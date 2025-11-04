#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_ac4k_op_soft_max(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_soft_max_back(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
