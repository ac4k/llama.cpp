#include "common.cuh"

#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

void ggml_ac4k_op_diag_mask_inf(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
