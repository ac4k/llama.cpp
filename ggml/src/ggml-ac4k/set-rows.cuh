#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_ac4k_op_set_rows(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
