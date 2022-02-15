#pragma once

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <string>

class CublasLtMMAlgoMap {
public:
    CublasLtMMAlgoMap() { }
    void GetAlgo(cublasLtMatmulAlgo_t& algo, const cudaDeviceProp& device_prop, int batch_count, int m, int n, int k, cublasLtOrder_t weight_order);

private:
    struct CublasLtMatmulAlgoInfo {
        int algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize;
        //in cublasLt >= 11.0
        int stages;
        float exec_time;
    };

    std::map<std::string, CublasLtMatmulAlgoInfo> best_algos_;
};

void cublasLt_MatMul_int8(
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
    int batchCount, int m, int n, int k,
    const float* alpha,
    const int8_t* A, int64_t batch_stride_A,
    const int8_t* B, int64_t batch_stride_B,
    const float* beta,
    const int8_t* C, int64_t ldc, // when ldc == 0, it will be (row) broadcast, otherwise ignored
    int8_t* D, int64_t batch_stride_D,
    int weight_order, /* ORDER_COL32_2R_4R4 or CUBLASLT_ORDER_COL4_4R2_8C */
    const CublasLtMMAlgoMap* algo_map,
    const cudaDeviceProp& prop);
