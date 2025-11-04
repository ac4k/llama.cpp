#include "common.cuh"

void ggml_ac4k_op_repeat(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
void ggml_ac4k_op_add(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
void ggml_ac4k_op_sub(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
void ggml_ac4k_op_mul(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
void ggml_ac4k_op_div(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_repeat_back(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_fused_add(ggml_backend_ac4k_context & ctx, ggml_tensor * dst, int n_fuse);
