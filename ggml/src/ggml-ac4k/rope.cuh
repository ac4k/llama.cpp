#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_ac4k_op_rope(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_rope_back(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
