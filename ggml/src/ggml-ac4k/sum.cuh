#include "common.cuh"

void sum_f32_cuda(ggml_ac4k_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream);

void ggml_ac4k_op_sum(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
