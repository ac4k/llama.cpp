#include "common.cuh"

#define CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void ggml_ac4k_op_timestep_embedding(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
