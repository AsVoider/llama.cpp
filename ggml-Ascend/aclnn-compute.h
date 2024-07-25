#ifndef COMPUTE_H
#define COMPUTE_H

#include <cstdint>
#include <vector>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "ggml.h"
#include "common.h"

void ggml_ascend_get_rows(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_cpy(ggml_backend_ascend_context &ctx, ggml_tensor *src, ggml_tensor *dst);

void ggml_ascend_dup(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_add(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_mul(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_silu(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_rms_norm(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_soft_max(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_soft_max_new(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_rope(ggml_backend_ascend_context &ctx, ggml_tensor *dst);

void ggml_ascend_mul_mat(ggml_backend_ascend_context &ctx, ggml_tensor *src0, ggml_tensor *src1, ggml_tensor *dst);

void ggml_ascend_silu_test(int64_t lens, int64_t width, float* data, int32_t deviceId, aclrtStream stream);

void ggml_ascend_cpy_test(int64_t* ne, float* data, int32_t deviceId, aclrtStream stream);

void ggml_ascend_dup_test(int64_t* ne, float* data, int32_t deviceId, aclrtStream stream);

void ggml_ascend_add_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream);

void ggml_ascend_mul_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream);

void ggml_ascend_get_rows_test(int64_t*ne1, int64_t*ne2, float* data1, int64_t* data2, int32_t deviceId, aclrtStream stream);

void ggml_ascend_soft_max_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, float scale, aclrtStream stream);
#endif