#include "common.cuh"

void ggml_ac4k_op_norm(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_group_norm(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_rms_norm(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_rms_norm_fused(ggml_backend_ac4k_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_ac4k_op_rms_norm_fused_add(ggml_backend_ac4k_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_ac4k_op_rms_norm_back(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_op_l2_norm(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
