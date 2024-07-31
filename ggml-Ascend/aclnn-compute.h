#ifndef COMPUTE_H
#define COMPUTE_H

#include <cstdint>
#include <vector>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_argsort.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_gelu.h>
#include <aclnnop/aclnn_hardsigmoid.h>
#include <aclnnop/aclnn_hardswish.h>
#include <aclnnop/aclnn_leaky_relu.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_tanh.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_rms_norm.h>
#include "aclnnop/aclnn_softmax.h"
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_roll.h>
#include <aclnnop/aclnn_index_fill_tensor.h>
#include <aclnnop/aclnn_matmul.h>

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