#include "common.cuh"

void ggml_ac4k_mul_mat_vec_f(ggml_backend_ac4k_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_ac4k_mm_fusion_args_host * fusion = nullptr);

void ggml_ac4k_op_mul_mat_vec_f(
    ggml_backend_ac4k_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_ac4k_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, int64_t ne11);
