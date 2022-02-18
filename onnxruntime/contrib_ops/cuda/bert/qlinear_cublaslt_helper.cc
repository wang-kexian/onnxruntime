#include "qlinear_cublaslt_helper.h"

void cublasLtMatMulInt8SetupAlgo(cublasLtMatmulAlgo_t& algo, cublasLtOrder_t weight_order, int algoId, int swizzle,
                                 int customOption, int tile, int splitK_val, int reductionScheme, int stages) {
  cublasLtMatmulAlgoInit(cublasLt_handle, compute_type, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
}

void CublasLtMMAlgoMap::GetAlgo(cublasLtMatmulAlgo_t& algo, const cudaDeviceProp& /*device_prop*/,
                                int batch_count, int m, int n, int k,
                                cublasLtOrder_t weight_order) {
  std::sstream ss;
  ss << batch_count << "-" << m << "_" << n << "_" << k << "_" << layout;
  std::string mark = ss.str();
  if (algo_map_.find(mark) != algo_map_.end() && algo_map_[mark].workspaceSize == 0) {
    const auto& algo_info = algo_map_[mark];
    cublasLtMatMulInt8SetupAlgo(algo, weight_order, algo_info.algoId, algo_info.swizzle,
                                algo_info.customOption, algo_info.tile, algo_info.reductionScheme, algo_info.stages);
  } else {
    int algoId = (weight_order == CUBLASLT_ORDER_COL4_4R2_8C) ? 6 : 7 /* CUBLASLT_ORDER_COL32_2R_4R4 */;
    int stages = (weight_order == CUBLASLT_ORDER_COL4_4R2_8C) ? 13 : 15 /* CUBLASLT_ORDER_COL32_2R_4R4 */;
    cublasLtMatMulInt8SetupAlgo(algo, weight_order, algoId, 0, 0, 20, 0, 0, stages);
  }
}

static int64_t
CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order) {
  switch (order) {
    case CUBLASLT_ORDER_ROW:
      return cols;
    case CUBLASLT_ORDER_COL:
      return rows;
    case CUBLASLT_ORDER_COL32:
      return 32 * rows;
    case CUBLASLT_ORDER_COL4_4R2_8C:
      return 32 * ((rows + 8 - 1) / 8) * 8;
    case CUBLASLT_ORDER_COL32_2R_4R4:
      return 32 * ((rows + 32 - 1) / 32) * 32;
    default:
      return 0;
  }
}

void cublasLt_MatMul_int8(cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                          int batchCount, int m, int n, int k,
                          const float* alpha,
                          const int8_t* A, int64_t batch_stride_A,
                          const int8_t* B, int64_t batch_stride_B,
                          const float* beta,
                          const int8_t* C, int64_t ldc,
                          int8_t* D, int64_t batch_stride_D,
                          cublasLtOrder_t weight_order, /* ORDER_COL32_2R_4R4 or CUBLASLT_ORDER_COL4_4R2_8C */
                          const CublasLtMMAlgoMap* algo_map,
                          const cudaDeviceProp& prop) {
  cublascompute_type_t compute_type = CUBLAS_COMPUTE_32I;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasLtmatmul_desc_t matmul_desc;
  cublasLtMatrixLayout_t desc_A, desc_B, desc_C, desc_D;
  desc_A = desc_B = desc_C = desc_D = nullptr;
  cublasLtOrder_t order_ACD = CUBLASLT_ORDER_COL32;
  cublasOperation_t transpose_B = CUBLAS_OP_T;

  int lda = 32 * m;
  int ldb = (weight_order == ORDER_COL32_2R_4R4) ? (32 * ((n + 32 - 1) / 32) * 32) : (32 * ((n + 8 - 1) / 8) * 8);
  int ldd = 32 * m;
  if (ldc != 0) ldc = ldd;
  int64_t batch_stride_C = (ldc == 0 ? 0LL : batch_stride_D);

  cublasLtmatmul_descCreate(&matmul_desc, compute_type, scale_type);
  cublasLtmatmul_descSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_B, sizeof(cublasOperation_t));
  cublasLtmatmul_descSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
  cublasLtMatrixLayoutCreate(&desc_A, CUDA_R_8I, m, k, lda);
  cublasLtMatrixLayoutSetAttribute(desc_A, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ACD, sizeof(order_ACD));
  cublasLtMatrixLayoutCreate(&desc_B, CUDA_R_8I, n, k, ldb);
  cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_ORDER, &weight_order, sizeof(weight_order));
  cublasLtMatrixLayoutCreate(&desc_C, CUDA_R_8I, m, n, ldc);
  cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ACD, sizeof(order_ACD));
  cublasLtMatrixLayoutCreate(&desc_D, CUDA_R_8I, m, n, ldd);
  cublasLtMatrixLayoutSetAttribute(desc_D, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_ACD, sizeof(order_ACD));
  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(desc_A, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(desc_A, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_A, sizeof(batch_stride_A));
    cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_B, sizeof(batch_stride_B));
    cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_C, sizeof(batch_stride_C));
    cublasLtMatrixLayoutSetAttribute(desc_D, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(desc_D, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_D, sizeof(batch_stride_D));
  }
  // get algo
  cublasLtMatmulAlgo_t algo;
  if (algo_map) {
    algo_map->GetAlgo(algo, prop, batchCount, m, n, k, weight_order);
  }

  cublasLtMatmul(cublasLt_handle, matmul_desc,
                 &alpha, A, desc_A, B, desc_B,
                 &beta, C, (C == D ? desc_D : desc_C), D, desc_D,
                 (algo_map != nullptr) ? (&algo) : (nullptr), nullptr, 0,  // algo, workspace, workspace_size
                 stream);

  cublasLtMatrixLayoutDestroy(desc_D);
  cublasLtMatrixLayoutDestroy(desc_C);
  cublasLtMatrixLayoutDestroy(desc_B);
  cublasLtMatrixLayoutDestroy(desc_A);
  cublasLtmatmul_descDestroy(matmul_desc);
}
