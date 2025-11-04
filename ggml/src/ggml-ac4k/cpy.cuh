#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_ac4k_cpy(ggml_backend_ac4k_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_ac4k_dup(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
