#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status QOrderQuantize(cudaStream_t stream, int64_t n, const T* src, int8_t* dst, float scale);

template <typename T>
Status QOrderDequantize(cudaStream_t stream, int64_t n, const int8_t* src, T* dst, float scale);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
