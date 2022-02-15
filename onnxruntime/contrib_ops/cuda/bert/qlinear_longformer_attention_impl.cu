
#include "qlinear_longformer_attention_impl.h"
#include "qlinear_cublaslt_helper.h"


// Launch the softmax kernel for non compact memory.
bool launchSoftmaxFastKernel(
    cudaStream_t stream,
    cublasHandle_t cublas,
    void* workspace,              // softmax space
    const void* q,                // transposed Q with shape (B, N, S, H)
    const void* k,                // transposed K with shape (B, N, S, H)
    const void* v,                // transposed V with shape (B, N, S, H)
    const void* attention_mask,   // attention mask with shape (B, S), with value 0.0 not masked, and -10000.0 masked.
    const void* global_q,         // Q for global tokens with shape (B, N, S, H)
    const void* global_k,         // K for global tokens with shape (B, N, S, H)
    const void* global_v,         // V for global tokens with shape (B, N, S, H)
    const int* global_attention,  // global attention with shape (B, S), with value 0 for local attention and 1 for global attention.
    const int* global_index,      // Global index with shape (B, S)
    const int* batch_global_num,  // Number of global tokens per batch with shape (B, 1)
    void* pinned_buffer,          // Pinned memory in CPU. Number of global tokens per batch with shape (B, 1)
    void* output,                 // output with shape (B, N, S, H)
    float scaler,                 // scalar
    int batch_size,               // batch size
    int sequence_length,          // sequence length
    int num_heads,                // number of heads
    int head_size,                // hidden size per head
    int attention_window,         // one sided windows size
    size_t element_size) {        // size of element: 2 for half, and 4 for float

  bool is_fp16 = (element_size == 2);
  void* scratch1 = reinterpret_cast<char*>(workspace);
  void* scratch2 = reinterpret_cast<char*>(scratch1) + GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length, sequence_length);

  // setup shared parameters for two strided batched matrix multiplies
  cudaDataType_t Atype;
  cudaDataType_t Btype;
  cudaDataType_t Ctype;
  cudaDataType_t resultType;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

  __half one_fp16, zero_fp16;
  float one_fp32, zero_fp32;
  void *alpha, *beta_0, *beta_1;

    one_fp16 = __float2half(1.f);
    zero_fp16 = __float2half(0.f);
    alpha = static_cast<void*>(&one_fp16);
    beta_0 = static_cast<void*>(&zero_fp16);
    beta_1 = static_cast<void*>(&one_fp16);
    Atype = CUDA_R_16F;
    Btype = CUDA_R_16F;
    Ctype = CUDA_R_16F;
    resultType = CUDA_R_16F;
    algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  // Strided batch matrix multiply
  //    qk = q * k^T
  // Shapes: q and k = B x N x S x H, qk = B x N x S x S
  // Convert col-major to row-major by swapping q and k in Gemm

  // Local attention part
  // S x S is calculated using sliding block WxW (W is one sided window size) like the following:
  //   [W][W]
  //   [W][W][W]
  //      [W][W][W]
  //         [W][W]
  // The first and last rows have 2 blocks, and the remaining has 3 blocks per row.
  // The calculation are splited into 3 parts. Firstly, fill the middle rows,  then the first row and finally the last row.
  // The results are stored in scratch1.

  int w = attention_window;
  int x_offset = num_heads * sequence_length * head_size;
  int y_offset = num_heads * sequence_length * sequence_length;
  int last_block = (sequence_length / w) - 1;
  int strideA = sequence_length * head_size;
  int strideB = sequence_length * head_size;
  int strideC = sequence_length * sequence_length;

  // When S == 2W, there is no middle rows of blocks:
  //   [W][W]
  //   [W][W]
  // We can use normal matrix multiplication in this case.
  if (sequence_length == 2 * w) {
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     sequence_length,
                                     sequence_length,
                                     head_size,
                                     alpha,
                                     k,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     q,
                                     Btype,
                                     head_size,
                                     sequence_length * head_size,
                                     beta_0,
                                     scratch1,
                                     Ctype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  } else {  // sequence_length > 2 * w
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_heads; ++j) {
        void* q_head = (char*)q + (i * x_offset + j * sequence_length * head_size + w * head_size) * element_size;
        void* k_head = (char*)k + (i * x_offset + j * sequence_length * head_size) * element_size;
        void* qk_head = (char*)scratch1 + (i * y_offset + j * sequence_length * sequence_length + w * sequence_length) * element_size;
        int count = (sequence_length - 2 * w) / w;
        CHECK(cublasGemmStridedBatchedEx(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         3 * w,                    // m
                                         w,                        // n
                                         head_size,                // k
                                         alpha,                    // alpha
                                         k_head,                   // A
                                         Atype,                    // A type
                                         head_size,                // lda
                                         w * head_size,            // strideA
                                         q_head,                   // B
                                         Btype,                    // B type
                                         head_size,                // ldb
                                         w * head_size,            // strideB
                                         beta_0,                   // beta
                                         qk_head,                  // C
                                         Ctype,                    // C type
                                         sequence_length,          // ldc
                                         sequence_length * w + w,  // strideC
                                         count,                    // batch count
                                         resultType,
                                         algo));
      }
    }

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     2 * w,                   // m
                                     w,                       // n
                                     head_size,               // k
                                     alpha,                   // alpha
                                     k,                       // A
                                     Atype,                   // A type
                                     head_size,               // lda
                                     strideA,                 // strideA
                                     q,                       // B
                                     Btype,                   // B type
                                     head_size,               // ldb
                                     strideB,                 // strideB
                                     beta_0,                  // beta
                                     scratch1,                // C
                                     Ctype,                   // C type
                                     sequence_length,         // ldc
                                     strideC,                 // strideC
                                     batch_size * num_heads,  // batch count
                                     resultType,
                                     algo));

    void* q_head = (char*)q + (last_block * w * head_size) * element_size;
    void* k_head = (char*)k + ((last_block - 1) * w * head_size) * element_size;
    void* qk_head = (char*)scratch1 + (last_block * w * sequence_length + (last_block - 1) * w) * element_size;
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     2 * w,
                                     w,
                                     head_size,
                                     alpha,
                                     k_head,
                                     Atype,
                                     head_size,
                                     strideA,
                                     q_head,
                                     Btype,
                                     head_size,
                                     strideB,
                                     beta_0,
                                     qk_head,
                                     Ctype,
                                     sequence_length,
                                     strideC,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  }

  const int* batch_global_count = reinterpret_cast<const int*>(pinned_buffer);
  // Global attention part
  for (int i = 0; i < batch_size; ++i) {
    if (batch_global_count[i] > 0) {
      void* q_batch = (char*)q + (i * x_offset) * element_size;
      void* k_batch = (char*)k + (i * x_offset) * element_size;
      void* qk_batch = (char*)scratch1 + (i * y_offset) * element_size;
      // Local tokens attending global tokens
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       batch_global_count[i],
                                       sequence_length,
                                       head_size,
                                       alpha,
                                       k_batch,
                                       Atype,
                                       head_size,
                                       strideA,
                                       q_batch,
                                       Btype,
                                       head_size,
                                       strideB,
                                       beta_0,
                                       qk_batch,
                                       Ctype,
                                       sequence_length,
                                       strideC,
                                       num_heads,
                                       resultType,
                                       algo));

      void* global_q_batch = (char*)global_q + (i * num_heads * sequence_length * head_size) * element_size;
      void* global_k_batch = (char*)global_k + (i * x_offset) * element_size;
      int strideB_global = sequence_length * head_size;

      // Global tokens attending everything
      // This GEMMs need to be last to make sure all global token entries are re-written.
      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       sequence_length,
                                       batch_global_count[i],
                                       head_size,
                                       alpha,
                                       global_k_batch,
                                       Atype,
                                       head_size,
                                       strideA,
                                       global_q_batch,
                                       Btype,
                                       head_size,
                                       strideB_global,
                                       beta_0,
                                       qk_batch,
                                       Ctype,
                                       sequence_length,
                                       strideC,
                                       num_heads,
                                       resultType,
                                       algo));
    }
  }

  int dim0 = sequence_length * num_heads;
  int dim1 = sequence_length;
  void* softmax_out = scratch2;

  const int blockSize = 64;
  const int gridSize = batch_size * num_heads * sequence_length;
  if (is_fp16) {
    LongformerSoftmaxFastKernel<__half, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_index,
        batch_global_num,
        static_cast<const __half*>(scratch1),
        static_cast<const __half*>(attention_mask),
        static_cast<__half*>(softmax_out), scaler, dim0, dim1, attention_window);
  } else {
    LongformerSoftmaxFastKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
        global_attention,
        global_index,
        batch_global_num,
        static_cast<const float*>(scratch1),
        static_cast<const float*>(attention_mask),
        static_cast<float*>(softmax_out), scaler, dim0, dim1, attention_window);
  }

  // Run the matrix multiply: output = softmax_out * v
  //   softmax_out: B x N x S x S
  //             v: B x N x S x H
  //      attn_out: B x N x S x H
  // Calculation uses full Gemm (S == 2W) or sliding blocks (S > 2W) in a way similar to local attention part.

  if (sequence_length == 2 * w) {
    // convert col-major to row-major by swapping softmax_out and v
    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,
                                     sequence_length,
                                     sequence_length,
                                     alpha,
                                     v,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     softmax_out,
                                     Btype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     beta_0,
                                     output,
                                     Ctype,
                                     head_size,
                                     sequence_length * head_size,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  } else {  // sequence_length > 2 * w
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_heads; ++j) {
        void* v_head = (char*)v + (i * x_offset + j * head_size * sequence_length) * element_size;
        void* prob_head = (char*)softmax_out + (i * y_offset + j * sequence_length * sequence_length + w * sequence_length) * element_size;
        void* out_head = (char*)output + (i * x_offset + j * head_size * sequence_length + w * head_size) * element_size;
        int count = (sequence_length - 2 * w) / w;
        CHECK(cublasGemmStridedBatchedEx(cublas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         head_size,
                                         w,
                                         3 * w,
                                         alpha,
                                         v_head,
                                         Atype,
                                         head_size,
                                         w * head_size,
                                         prob_head,
                                         Btype,
                                         sequence_length,
                                         sequence_length * w + w,
                                         beta_0,
                                         out_head,
                                         Ctype,
                                         head_size,
                                         w * head_size,
                                         count,
                                         resultType,
                                         algo));
      }
    }

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,
                                     w,
                                     2 * w,
                                     alpha,
                                     v,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     softmax_out,
                                     Btype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     beta_0,
                                     output,
                                     Ctype,
                                     head_size,
                                     sequence_length * head_size,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));

    void* v_head = (char*)v + (last_block - 1) * w * head_size * element_size;
    void* prob_head = (char*)softmax_out + (sequence_length * last_block * w + (last_block - 1) * w) * element_size;
    void* out_head = (char*)output + last_block * w * head_size * element_size;

    CHECK(cublasGemmStridedBatchedEx(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     head_size,
                                     w,
                                     2 * w,
                                     alpha,
                                     v_head,
                                     Atype,
                                     head_size,
                                     sequence_length * head_size,
                                     prob_head,
                                     Btype,
                                     sequence_length,
                                     sequence_length * sequence_length,
                                     beta_0,
                                     out_head,
                                     Ctype,
                                     head_size,
                                     sequence_length * head_size,
                                     batch_size * num_heads,
                                     resultType,
                                     algo));
  }

  for (int i = 0; i < batch_size; ++i) {
    if (batch_global_count[i] > 0) {
      int glob_longdim_mm = (last_block - 1) * w;

      void* v_head = (char*)v + (i * x_offset) * element_size;
      void* prob_head = (char*)softmax_out + (i * y_offset + 2 * w * sequence_length) * element_size;
      void* out_head = (char*)output + (i * x_offset + 2 * w * head_size) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,
                                       glob_longdim_mm,
                                       batch_global_count[i],
                                       alpha,
                                       v_head,
                                       Atype,
                                       head_size,
                                       sequence_length * head_size,
                                       prob_head,
                                       Btype,
                                       sequence_length,
                                       sequence_length * sequence_length,
                                       beta_1,
                                       out_head,
                                       Ctype,
                                       head_size,
                                       sequence_length * head_size,
                                       num_heads,
                                       resultType,
                                       algo));

      // Global tokens
      v_head = (char*)global_v + (i * x_offset) * element_size;
      prob_head = (char*)softmax_out + (i * y_offset) * element_size;
      out_head = (char*)output + (i * x_offset) * element_size;

      CHECK(cublasGemmStridedBatchedEx(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       head_size,
                                       batch_global_count[i],
                                       sequence_length,  // Re-write entries completely
                                       alpha,
                                       v_head,
                                       Atype,
                                       head_size,
                                       sequence_length * head_size,
                                       prob_head,
                                       Btype,
                                       sequence_length,
                                       sequence_length * sequence_length,
                                       beta_0,    // Use beta=0 to overwrite
                                       out_head,  // Here assumes global tokens are at the beginning of sequence.
                                       Ctype,
                                       head_size,
                                       sequence_length * head_size,
                                       num_heads,
                                       resultType,
                                       algo));
    }
  }

  return true;
}


bool QLinearLongformerQkvToContext(
    const cudaDeviceProp& device_prop,
    cublasLtHandle_t& cublasLt, cudaStream_t stream,
    const int batch_size,       // B
    const int sequence_length,  // S
    const int num_heads,        // N
    const int head_size,        // H
    const int window,           // W
    const int max_num_global,   // G
    const int8_t* input, float scale_input,
    const float* attention_mask,
    const int8_t* global_input, float scale_global_input,
    const int* global_attention,
    const int* global_index, const int* batch_global_num,
    void* pinned_buffer,
    void* workspace,
    int8_t* output,
    size_t softmax_workspace_size,
    bool use_fast_kernel) {
  uint8_t* qkv = reinterpret_cast<uint8_t*>((char*)workspace + softmax_workspace_size);

  // Number of elements in Q, K, V, Global_Q, Global_K or Global_V are same: BxNxSxH
  const int elements = batch_size * num_heads * sequence_length * head_size;

  const int max_threads_per_block(device_prop.maxThreadsPerBlock);

  // Input should be BxSx3xNxH => qkv: 3xBxNxSxH
  if (!LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size, num_heads, max_threads_per_block, false, input, qkv)) {
    return false;
  }

  // Input 'global_input' should be BxSx3xNxH => global_qkv: 3xBxNxSxH
  T* global_qkv = qkv + 3 * elements;

  // When there is no global token, no need to process global Q, K and V
  if (max_num_global > 0 && nullptr != global_input) {
    if (!LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size, num_heads, max_threads_per_block, false, global_input, global_qkv)) {
      return false;
    }
  }

  // Now qkv has Q, K, V: each has size BxNxSxH
  const T* q = qkv;
  const T* k = q + elements;
  const T* v = k + elements;

  const T* global_q = global_qkv;
  const T* global_k = global_q + elements;
  const T* global_v = global_k + elements;

  // Q*K' are scaled by 1/sqrt(H)
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));

  T* temp_output = qkv;  // Q will be overwritten

  if (use_fast_kernel) {
    if (!launchSoftmaxFastKernel(
            stream,
            cublas,
            workspace,         // softmax space
            q,                 // transposed Q with shape (B, N, S, H)
            k,                 // transposed K with shape (B, N, S, H)
            v,                 // transposed V with shape (B, N, S, H)
            attention_mask,    // attention mask with shape (B, S), with value 0.0 not masked, and -10000.0 masked.
            global_q,          // Q for global tokens with shape (B, N, S, H)
            global_k,          // K for global tokens with shape (B, N, S, H)
            global_v,          // V for global tokens with shape (B, N, S, H)
            global_attention,  // global attention with shape (B, S), with value 0 for local attention and 1 for global attention.
            global_index,      // Global index with shape (B, S)
            batch_global_num,  // Number of global tokens per batch with shape (B, 1)
            pinned_buffer,     // Pinned memory in CPU. Number of global tokens per batch with shape (B, 1)
            temp_output,       // output with shape (B, N, S, H)
            rsqrt_head_size,   // scalar
            batch_size,        // batch size
            sequence_length,   // sequence length
            num_heads,         // number of heads
            head_size,         // hidden size per head
            window,            // Half (one-sided) window size
            element_size)) {
      return false;
    }
  } else {
    assert(max_num_global <= window);
    if (!launchSoftmaxKernel(
            stream,
            cublas,
            workspace,         // softmax space
            q,                 // Transposed Q with shape B x N x S x H
            k,                 // Transposed K with shape B x N x S x H
            v,                 // Transposed V with shape B x N x S x H
            attention_mask,    // Attention mask flags with shape B x S. Value -10000.0 means masked, and 0.0 not mased.
            global_q,          // Transposed global Q with shape B x N x S x H.
            global_k,          // Transposed global K with shape B x N x S x H
            global_v,          // Transposed global V with shape B x N x S x H
            global_attention,  // Global attention flags with shape B x S
            global_index,      // Global index with shape B x S
            batch_global_num,  // Number of global token per batch with shape B x 1
            pinned_buffer,     // Pinned Memory Buffer
            temp_output,       // Output with shape B x N x S x H
            rsqrt_head_size,   // Scaler
            batch_size,        // Batch size
            sequence_length,   // Sequence length
            num_heads,         // Number of attention heads
            head_size,         // Hidden size per head
            window,            // Half (one-sided) window size
            element_size)) {
      return false;
    }
  }


  // The temp_output is BxNxSxH, transpose it to final output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, max_threads_per_block, false, temp_output, output);
}