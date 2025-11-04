#include "common.cuh"

#define CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE 256

void ggml_ac4k_cross_entropy_loss(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);

void ggml_ac4k_cross_entropy_loss_back(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
