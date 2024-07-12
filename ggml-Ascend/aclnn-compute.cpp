#include "aclnn-compute.h"
#include "aclnn-add.h"
#include "aclnn-mul.h"
#include "aclnn-cpy.h"
#include "aclnn-unary.h"

void ggml_ascend_get_rows(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    std::vector<int64_t> selfShape = {1, 1, ne01 * ne02 * ne03, ne00};
    std::vector<int64_t> indexShape = {ne10 * ne11 * ne12};
    std::vector<int64_t> outShape = {1, 1, ne10 * ne11 * ne12, ne00};

    
}

void ggml_ascend_add(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    std::vector<int64_t> selfShape = {ne03, ne02, ne01, ne00};
    std::vector<int64_t> otherShape = {ne13, ne12, ne11, ne10};
    std::vector<int64_t> outShape = {
        (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
        (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
        (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
        (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
    };

    int ret = aclnn_add_func(src0->data, src1->data, dst->data,
                            selfShape, otherShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
}

void ggml_ascend_mul(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    std::vector<int64_t> selfShape = {ne03, ne02, ne01, ne00};
    std::vector<int64_t> otherShape = {ne13, ne12, ne11, ne10};
    std::vector<int64_t> outShape = {
        (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
        (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
        (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
        (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
    };

    int ret = aclnn_mul_func(src0->data, src1->data, dst->data,
                            selfShape, otherShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
}

void ggml_ascend_cpy(ggml_backend_ascend_context &ctx, ggml_tensor *src, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    std::vector<int64_t> srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    std::vector<int64_t> dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

    int ret = aclnn_cpy_func(dst->data, src->data,
                            dstShape, srcShape,
                            ggml_to_acl_map[dst->type], ggml_to_acl_map[src->type],
                            stream);
}

void ggml_ascend_dup(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src = dst->src[0];
    ggml_ascend_cpy(ctx, src, dst);
}

void ggml_ascend_silu(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src = dst->src[0];
    aclrtStream stream = ctx.stream();

    std::vector<int64_t> srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    std::vector<int64_t> dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

    int ret = aclnn_silu_func(src->data, dst->data,
                            srcShape, dstShape,
                            ggml_to_acl_map[src->type], ggml_to_acl_map[dst->type],
                            stream);
}