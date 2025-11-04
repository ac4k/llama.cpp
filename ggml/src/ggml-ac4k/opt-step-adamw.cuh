#include "common.cuh"

#define CUDA_OPT_STEP_ADAMW_BLOCK_SIZE 256

void ggml_ac4k_opt_step_adamw(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
