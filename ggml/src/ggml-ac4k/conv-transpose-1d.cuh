#include "common.cuh"

#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

void ggml_ac4k_op_conv_transpose_1d(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
