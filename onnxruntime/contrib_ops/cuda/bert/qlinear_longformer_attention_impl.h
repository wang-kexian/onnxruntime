// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "qlinear_cublaslt_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool QLinearLongformerQkvToContext(
    const cudaDeviceProp& device_prop,
    cublasLtHandle_t& cublas, cudaStream_t stream,
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
    bool use_fast_kernel);

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
    size_t element_size);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
