#ifndef COMPUTE_H
#define COMPUTE_H

#include <cstdint>
#include <vector>

void ggml_ascend_get_rows(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_cpy(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_dup(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_add(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_mul(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_silu(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_rms_norm(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_soft_max(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_rope(ggml_backend_ascned_context &ctx, ggml_tensor *dst);

void ggml_ascend_mul_mat(ggml_backend_ascned_context &ctx, ggml_tensor *dst);


#endif