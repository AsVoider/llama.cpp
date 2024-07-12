#include "aclnn-compute.h"
#include "aclnn-add.h"

aclDataType ggml_to_acl_map[GGML_TYPE_COUNT] = {
    ACL_FLOAT,        // GGML_TYPE_F32
    ACL_FLOAT16,      // GGML_TYPE_F16
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_0
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_1
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_2 (removed)
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_3 (removed)
    ACL_DT_UNDEFINED, // GGML_TYPE_Q5_0
    ACL_DT_UNDEFINED, // GGML_TYPE_Q5_1
    ACL_DT_UNDEFINED, // GGML_TYPE_Q8_0
    ACL_DT_UNDEFINED, // GGML_TYPE_Q8_1
    ACL_DT_UNDEFINED, // GGML_TYPE_Q2_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q3_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q5_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q6_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q8_K
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ2_XXS
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ2_XS
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ3_XXS
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ1_S
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ4_NL
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ3_S
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ2_S
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ4_XS
    ACL_INT8,         // GGML_TYPE_I8
    ACL_INT16,        // GGML_TYPE_I16
    ACL_INT32,        // GGML_TYPE_I32
    ACL_INT64,        // GGML_TYPE_I64
    ACL_DOUBLE,       // GGML_TYPE_F64
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ1_M
    ACL_BF16          // GGML_TYPE_BF16
};


void ggml_ascend_add(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

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
                            ctx.stream());
}