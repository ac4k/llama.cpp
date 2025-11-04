#include "common.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);
void ggml_ac4k_op_sum_rows(ggml_backend_ac4k_context & ctx, ggml_tensor * dst);
