#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <string>

template <typename T, typename ScaleT, int items_per_thread = 8>
__global__
void QOrderedQuantizeKernel(int64_t n, const T* src, int8_t* dst, const ScaleT* scale_ptr) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x * items_per_thread  + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < items_per_thread; i++) {
      idx += blockDim.x;
      if (idx < n) {
          T v = scale * src[idx];
          v = min(v, T{127.0});
          v = max(v, T{-128.0});
          dst[idx] = (int8_t)v;
      }
  }
}

template <typename T>
bool QOrderedQuantize(int64_t n, const T* src, int8_t* dst, T scale)
{
    QOrderedQuantizeKernel<<<(n + 4095)/(256 * 8), 256>>>(n, src, dst, scale);
}

template bool QOrderedQuantize(int64_t n, const half* src, int8_t* dst, half scale);
template bool QOrderedQuantize(int64_t n, const float* src, int8_t* dst, float scale);
