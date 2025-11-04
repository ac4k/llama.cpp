#include "common.cuh"

#define CUDA_CLAMP_BLOCK_SIZE 256

void ggml_ac4k_op_clamp(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
