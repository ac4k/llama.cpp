#include "common.cuh"

#define CUDA_COUNT_EQUAL_CHUNK_SIZE 128

void ggml_ac4k_count_equal(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
