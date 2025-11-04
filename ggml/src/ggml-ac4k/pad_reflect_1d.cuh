#include "common.cuh"

#define CUDA_PAD_REFLECT_1D_BLOCK_SIZE 256

void ggml_ac4k_op_pad_reflect_1d(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
