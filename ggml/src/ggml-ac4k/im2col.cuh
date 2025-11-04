#include "common.cuh"

#define CUDA_IM2COL_BLOCK_SIZE 256

void ggml_ac4k_op_im2col(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
void ggml_ac4k_op_im2col_3d(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
