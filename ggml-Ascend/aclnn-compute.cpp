#include "aclnn-compute.h"
#include "aclnn-comp.h"
#include "string.h"
#include "aclnn-add.h"
#include "aclnn-mul.h"
#include "aclnn-cpy.h"
#include "aclnn-unary.h"
#include "../../../../../Ascend/ascend-toolkit/8.0.RC2.alpha003/toolkit/toolchain/hcc/sysroot/usr/include/string.h"

void ggml_ascend_get_rows(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    aclnn_shape_t selfShape = {1, 1, ne01 * ne02 * ne03, ne00};
    aclnn_shape_t indexShape = {ne10 * ne11 * ne12};
    aclnn_shape_t outShape = {1, 1, ne10 * ne11 * ne12, ne00};
    aclnn_shape_t tmpHostData(ne10 * ne11 * ne12 * ne13, 0);
    aclnn_shape_t offset;
    for(int i = 0; i < ne11 * ne12 * ne13; i++) {
        offset.insert(offset.end(), ne10, i * ne01);
    }

    void* offsetDeviceAddr = nullptr;
    void* tmpDeviceAddr = nullptr;
    
    int ret = data_addr_malloc(indexShape, offset, &offsetDeviceAddr);
    ret = data_addr_malloc(indexShape, tmpHostData, &tmpDeviceAddr);

    ret = aclnn_add_func(src1->data, offsetDeviceAddr, tmpDeviceAddr,
                        indexShape, indexShape, indexShape,
                        ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type],
                        stream);

    ret = aclnn_get_rows_func(src0->data, tmpDeviceAddr, dst->data,
                            selfShape, indexShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);

    aclrtFree(offsetDeviceAddr);
    aclrtFree(tmpDeviceAddr);
}

void ggml_ascend_add(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    aclnn_shape_t selfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t otherShape = {ne13, ne12, ne11, ne10};
    aclnn_shape_t outShape = {
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

    aclnn_shape_t selfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t otherShape = {ne13, ne12, ne11, ne10};
    aclnn_shape_t outShape = {
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

    aclnn_shape_t srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    aclnn_shape_t dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

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

    aclnn_shape_t srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    aclnn_shape_t dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

    int ret = aclnn_silu_func(src->data, dst->data,
                            srcShape, dstShape,
                            ggml_to_acl_map[src->type], ggml_to_acl_map[dst->type],
                            stream);
}
