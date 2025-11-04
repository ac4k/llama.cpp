#include "common.cuh"

#define CUDA_ROLL_BLOCK_SIZE 256

void ggml_ac4k_op_roll(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
