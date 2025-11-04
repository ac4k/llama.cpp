#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

void ggml_ac4k_op_rwkv_wkv6(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_rwkv_wkv7(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
