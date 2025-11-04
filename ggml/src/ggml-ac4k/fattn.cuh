#include "common.cuh"

void ggml_ac4k_flash_attn_ext(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

bool ggml_ac4k_flash_attn_ext_supported(int device, const ggml_tensor * dst);
