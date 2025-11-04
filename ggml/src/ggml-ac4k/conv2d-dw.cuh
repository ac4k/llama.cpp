#pragma once
#include "common.cuh"

#define CUDA_CONV2D_DW_BLOCK_SIZE 256
void ggml_ac4k_op_conv2d_dw(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
