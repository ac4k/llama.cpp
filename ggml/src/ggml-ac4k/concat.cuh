#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_ac4k_op_concat(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
