#include "common.cuh"

#define CUDA_UPSCALE_BLOCK_SIZE 256

void ggml_ac4k_op_upscale(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
