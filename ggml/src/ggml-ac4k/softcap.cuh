#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_ac4k_op_softcap(ggml_backend_ac4k_context & ctx, ggml_tensor * dst, ggml_tensor * src);
