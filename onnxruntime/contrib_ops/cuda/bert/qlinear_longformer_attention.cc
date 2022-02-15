// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "longformer_attention.h"
#include "longformer_global_impl.h"
#include "longformer_attention_impl.h"
#include "transformer_cuda_common.h"
#include "transformer_common.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearLongformerAttention,
    kMSDomain,
    1,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    QLinearLongformerAttention);

QLinearLongformerAttention::QLinearLongformerAttention(const OpKernelInfo& info) : CudaKernel(info), LongformerAttentionBase(info) {
  use_compact_memory_ = ParseEnvironmentVariableWithDefault<bool>(longformer::kUseCompactMemory, false);
}

#define ScaleTensorValue(scale_name, idx)                          \
  const Tensor* tensor_##scale_name = context->Input<Tensor>(idx); \
  float##scale_name = *(tensor_##scale_name->template Data<float>())

Status QLinearLongformerAttention::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);
  const Tensor* global_weights = context->Input<Tensor>(4);
  const Tensor* global_bias = context->Input<Tensor>(5);
  const Tensor* global_attention = context->Input<Tensor>(6);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention->Shape()));
  ScaleTensorValue(scale_input, 7);
  ScaleTensorValue(scale_weight, 8);
  ScaleTensorValue(scale_bias, 9);
  ScaleTensorValue(scale_01, 10);
  ScaleTensorValue(scale_global_weight, 11);
  ScaleTensorValue(scale_global_bias, 12);
  ScaleTensorValue(scale_02, 13);

  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Output 0 - output     : (batch_size, sequence_length, hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  cublasHandle_t cublas = CublasHandle();
  cublasLtHandle_t cublasLt = cublasLtHandle();
  cudaStream_t stream = (cudaStream_t)GetComputeStream();

  // TODO: only calculate once per model.
  // Build Global Index
  auto global_index_buffer = GetScratchBuffer<int>(batch_size * sequence_length);
  auto batch_global_num_buffer = GetScratchBuffer<int>(batch_size);

  size_t global_scratch_bytes = GetGlobalScratchSize(batch_size, sequence_length);
  auto global_scratch_buffer = GetScratchBuffer<void>(global_scratch_bytes);

  BuildGlobalIndex(
      stream,
      global_attention->template Data<int>(),
      batch_size,
      sequence_length,
      global_index_buffer.get(),
      batch_global_num_buffer.get(),
      global_scratch_buffer.get(),
      global_scratch_bytes);

  // Copy batch_global_num to CPU
  size_t pinned_buffer_bytes = GetPinnedBufferSize(batch_size);
  auto pinned_buffer = AllocateBufferOnCPUPinned<void>(pinned_buffer_bytes);
  int* batch_global_num_pinned = reinterpret_cast<int*>(pinned_buffer.get());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(batch_global_num_pinned, batch_global_num_buffer.get(), batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream));

  // Create an event to make sure the async copy is finished before reading the data.
  AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();

  CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, stream));

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = hidden_size;

  size_t qkv_size = batch_size * sequence_length * 3 * hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(qkv_size);

  // input * weight + bias
  auto& device_prop = GetDeviceProp();
  float alpha_input = scale_input * scale_weight / scale_01;
  float beta_input = scale_bias;
  CUBLAS_RETURN_IF_ERROR(cublasLt_MatMul_int8(
      cublasLt, stream,
      1, m, n, k,
      &alpha_input,
      reinterpret_cast<const int8_t*>(input->template Data<int8_t>()), 0,
      reinterpret_cast<const int8_t*>(weights->template Data<int8_t>()), 0,
      &beta_input,
      reinterpret_cast<const int8_t*>(bias->template Data<int8_t>()), 0,
      reinterpret_cast<const int8_t*>(gemm_buffer.get()), 0,
      CUBLASLT_ORDER_COL4_4R2_8C,
      nullptr,
      device_prop));

  // Wait for async copy of batch_global_num
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));

  // Find the maximum number of global tokens in all batches
  int max_num_global = *std::max_element(batch_global_num_pinned, batch_global_num_pinned + batch_size);

  // Force to use fast kernel in two situations:
  // (1) global tokens > windows size. In that case, compact memory kernel cannot be used.
  // (2) sequence_length == 2 * attention_window. Use fast kernel to walk around parity issue of compact memory kernel.
  // In other case, we will choose according to user's environment variable setting (default is fast kernel).
  bool use_fast_kernel = (max_num_global > window_ || sequence_length == 2 * window_ || !use_compact_memory_);

  // Fully connection for global projection.
  // Note that Q only need handle global query tokens if we split GEMM to global Q/K/V separately.
  // When there is no global token, need not run glboal GEMM.
  auto global_gemm_buffer = GetScratchBuffer<T>(max_num_global > 0 ? qkv_size : 0);

  float alpha_global_input = scale_input * scale_global_weight / scale_02;
  float beta_global_input = scale_global_bias;
  if (max_num_global > 0) {
    CUBLAS_RETURN_IF_ERROR(cublasLt_MatMul_int8(
        cublasLt, stream,
        1, m, n, k,
        &alpha_global_input,
        reinterpret_cast<const int8_t*>(input->template Data<int8_t>()), 0,
        reinterpret_cast<const int8_t*>(global_weights->template Data<int8_t>()), 0,
        &beta_global_input,
        reinterpret_cast<const int8_t*>(global_bias->template Data<int8_t>()), 0,
        reinterpret_cast<const int8_t*>(global_gemm_buffer.get()), 0,
        CUBLASLT_ORDER_COL4_4R2_8C,
        nullptr,
        device_prop));
  }

  size_t workSpaceSize = GetLongformerAttentionWorkspaceSize(
      1, batch_size, num_heads_, head_size, sequence_length, max_num_global, window_, use_fast_kernel);
  auto workspace_buffer = GetScratchBuffer<void>(workSpaceSize);
  size_t softmax_workspace_size = GetLongformerSoftmaxWorkspaceSize(
      1, batch_size, num_heads_, sequence_length, window_, use_fast_kernel);

  if (!QLinearLongformerQkvToContext(
          device_prop, cublas, stream,
          batch_size, sequence_length, num_heads_, head_size, window_, max_num_global,
          reinterpret_cast<const int8_t*>(gemm_buffer.get()), alpha_input,
          reinterpret_cast<const float*>(mask->template Data<float>()),
          reinterpret_cast<const int8_t*>(global_gemm_buffer.get()), alpha_global_input,
          global_attention->template Data<int>(),
          global_index_buffer.get(),
          batch_global_num_buffer.get(),
          pinned_buffer.get(),
          workspace_buffer.get(),
          output->template MutableData<int8_t>(),
          softmax_workspace_size,
          use_fast_kernel)) {
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  this->AddDeferredReleaseCPUPtr(pinned_buffer.release());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
