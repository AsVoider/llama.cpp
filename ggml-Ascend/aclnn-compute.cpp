#include "aclnn-compute.h"
#include "aclnn-add.h"
#include "aclnn-mul.h"
#include "aclnn-cpy.h"
#include "aclnn-unary.h"
#include "aclnn-norm.h"
#include "aclnn-math.h"
#include "aclnn-rope.h"
#include <cstring>
#include <cmath>
#include <algorithm>

void ggml_ascend_get_rows(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    aclnn_shape_t selfShape = {1, 1, ne01 * ne02 * ne03, ne00};
    aclnn_shape_t indexShape = {ne10 * ne11 * ne12};
    aclnn_shape_t outShape = {1, 1, ne10 * ne11 * ne12, ne00};
    std::vector<int64_t> tmpHostData(ne10 * ne11 * ne12 * ne13, 0);
    std::vector<int64_t> offset;
    for(int i = 0; i < ne11 * ne12 * ne13; i++) {
        offset.insert(offset.end(), ne10, i * ne01);
    }

    void* offsetDeviceAddr = nullptr;
    void* tmpDeviceAddr = nullptr;
    
    int ret = data_addr_malloc(indexShape, offset, &offsetDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ret = data_addr_malloc(indexShape, tmpHostData, &tmpDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ret = aclnn_add_func(src1->data, offsetDeviceAddr, tmpDeviceAddr,
                        indexShape, indexShape, indexShape,
                        ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type],
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ret = aclnn_get_rows_func(src0->data, tmpDeviceAddr, dst->data,
                            selfShape, indexShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, return);

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
    CHECK_RET(ret == ACL_SUCCESS, return);
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
    CHECK_RET(ret == ACL_SUCCESS, return);
}

void ggml_ascend_cpy(ggml_backend_ascend_context &ctx, ggml_tensor *src, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    aclnn_shape_t srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    aclnn_shape_t dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

    int ret = aclnn_cpy_func(dst->data, src->data,
                            dstShape, srcShape,
                            ggml_to_acl_map[dst->type], ggml_to_acl_map[src->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, return);
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
    CHECK_RET(ret == ACL_SUCCESS, return);
}

void ggml_ascend_rms_norm(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    aclrtStream stream = ctx.stream();

    aclnn_shape_t xShape = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    aclnn_shape_t gammaShape = {src0->ne[0]};
    aclnn_shape_t yShape = xShape;
    aclnn_shape_t rstdShape = {src0->ne[0], 1};

    void* gammaDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    std::vector<float> gammaHostData(src0->ne[0], 1);
    std::vector<float> rstdHostData = {1, float(src0->ne[0])};

    int ret = data_addr_malloc(gammaShape, gammaHostData, &gammaDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(rstdShape, rstdHostData, &rstdDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    ret = aclnn_rms_norm_func(src0->data, gammaDeviceAddr, dst->data, rstdDeviceAddr,
                                xShape, gammaShape, yShape, rstdShape,
                                ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                eps, stream);
    CHECK_RET(ret == ACL_SUCCESS, return);

    aclrtFree(gammaDeviceAddr);
    aclrtFree(rstdDeviceAddr);
}

void ggml_ascend_soft_max(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {

}

void ggml_ascend_mul_mat(ggml_backend_ascend_context &ctx, ggml_tensor *src0, ggml_tensor *src1, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    aclnn_shape_t selfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t otherShape = {ne13, ne12, ne11, ne10};
    aclnn_shape_t outShape = {ne3, ne2, ne1, ne0};

    int ret = aclnn_mul_mat_func(src0->data, src1->data, dst->data,
                                selfShape, otherShape, outShape,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                                stream);
    CHECK_RET(ret == ACL_SUCCESS, return);
}

void ggml_ascend_rope(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {

    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    int64_t* ne = src0->ne;
    float freq_scale = dst->op_params[6];
    float freq_base = dst->op_params[5];
    int n_dims = dst->op_params[1];
    void* pos = src1->data;

    int64_t size = ne[0] * ne[1] * ne[2] * ne[3];

    float theta_scale_pow[ne[0] / 2];
    float theta_base[size];
    float theta[size];
    float sin_d[size];
    float cos_d[size];

    float theta_scale = pow(freq_base, -2.0 / n_dims);

    std::vector<float> powExpHostData(ne[0]/2, 0);
    std::vector<float> powOutData = powExpHostData;
    aclnn_shape_t powExpShape = {ne[0]/2, 1};
    aclnn_shape_t powOutShape = powExpShape;
    std::generate_n(powExpHostData.begin(), powExpHostData.size(), [&, index = 0]() mutable {
        return index++;
    });
    void* powExpDeviceAddr = nullptr;
    void* powOutDeviceAddr = nullptr;
    int ret = data_addr_malloc(powExpShape, powExpHostData, &powExpDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(powOutShape, powOutData, &powOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ret = aclnn_pow_scalar_tensor_func(theta_scale, powExpDeviceAddr, powOutDeviceAddr,
        powExpShape, powOutShape,
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        stream);
    CHECK_RET(ret == ACL_SUCCESS, return);

    void* mulOtherDeviceAddr = nullptr;
    ret = aclrtMalloc(&mulOtherDeviceAddr, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return);
    for (int i = 0; i < size; i += ne[0] / 2) {
        ret = aclrtMemcpy(mulOtherDeviceAddr + i * sizeof(float), ne[0] / 2 * sizeof(float), powOutDeviceAddr, ne[0] / 2 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return);
    }

    // std::vector<float> mulOtherHostData(size, 0);
    // std::generate_n(mulOtherHostData.begin(), size, [&, index = 0]() mutable {
    //     return theta_scale_pow[index++ % (ne[0]/2)];
    // });
    // for (int i=0; i< mulOtherHostData.size(); i++){
    //   std::cout << mulOtherHostData[i]<< " "; 
    // }
    // return;

    aclnn_shape_t mulSelfShape = {size, 1};
    aclnn_shape_t mulOtherShape = mulSelfShape;
    aclnn_shape_t mulOutShape = mulSelfShape;

    void* mulOutDeviceAddr = nullptr;
    void* mulSelfDeviceAddr = nullptr;
    std::vector<float> mulOutHostData(size, 0);
    ret = data_addr_malloc(mulOutShape, mulOutHostData, &mulOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ret = aclrtMalloc(&mulSelfDeviceAddr, size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return);
    for (int i = 0; i < size; i += ne[2]) {
        ret = aclrtMemcpy(mulSelfDeviceAddr + i * sizeof(int32_t), ne[2] * sizeof(int32_t), pos, ne[2] * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return);
    }

    // std::vector<float> mulSelfHostData(size, 0);
    // std::generate_n(mulSelfHostData.begin(), size, [&, index = 0]() mutable {
    //     return (float)pos[index++ % ne[2]];
    // });
    // for (int i=0; i< mulSelfHostData.size(); i++){
    //   std::cout << mulSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_mul_func(mulSelfDeviceAddr, mulOtherDeviceAddr, mulOutDeviceAddr,
                        mulSelfShape, mulOtherShape, mulOutShape,
                        ACL_INT32, ACL_FLOAT, ACL_FLOAT,
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, return);

    // ret = aclnnMulFunc(mulSelfShape, mulOtherShape, mulOutShape, mulSelfHostData, mulOtherHostData, mulOutHostData, theta_base, context, stream);

    void* mulsOutDeviceAddr = nullptr;
    void* mulsSelfDeviceAddr = mulOutDeviceAddr;
    aclnn_shape_t mulsSelfShape = mulSelfShape;
    aclnn_shape_t mulsOutShape = mulOutShape;
    std::vector<float> mulsOutHostData(size, 0);
    ret = data_addr_malloc(mulsOutShape, mulsOutHostData, &mulsOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    // std::vector<float> mulsSelfHostData(theta_base, theta_base+ size);

    // for (int i=0; i< mulsSelfHostData.size(); i++){
    //   std::cout << mulsSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_muls_func(mulsSelfDeviceAddr, mulsOutDeviceAddr, mulsSelfShape, mulsOutShape, ACL_FLOAT, ACL_FLOAT, freq_scale, stream);
    CHECK_RET(ret == ACL_SUCCESS, return);
    // ret = aclnnMulsFunc(mulsSelfHostData, mulsOutHostData, freq_scale, mulsSelfShape, mulsOutShape, theta, context, stream);

    void* sinSelfDeviceAddr = mulsOutDeviceAddr;
    void* sinOutDeviceAddr = nullptr;
    aclnn_shape_t sinSelfShape = {size, 1};
    aclnn_shape_t sinOutShape = {size, 1};
    std::vector<float> sinOutHostData(size, 0);
    ret = data_addr_malloc(sinOutShape, sinOutHostData, &sinOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    // std::vector<float> sinSelfHostData(theta, theta+ size);


    // for (int i=0; i< sinSelfHostData.size(); i++){
    //   std::cout << sinSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_sin_func(sinSelfDeviceAddr, sinOutDeviceAddr, sinSelfShape, sinOutShape, ACL_FLOAT, ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, return);
    // ret = aclnnSinFunc(sinSelfShape, sinShape, sinSelfHostData, sinHostData, sin_d, context, stream);

    void* cosSelfDeviceAddr = mulsOutDeviceAddr;
    void* cosOutDeviceAddr = nullptr;
    aclnn_shape_t cosSelfShape = {size, 1};
    aclnn_shape_t cosOutShape = {size, 1};
    std::vector<float> cosOutHostData(size, 0);
    ret = data_addr_malloc(cosOutShape, cosOutHostData, &cosOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    // std::vector<float> cosSelfHostData(theta, theta+ size);

    ret = aclnn_cos_func(cosSelfDeviceAddr, cosOutDeviceAddr, cosSelfShape, cosOutShape, ACL_FLOAT, ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, return);
    // ret = aclnnCosFunc(cosSelfShape, cosShape, cosSelfHostData, cosHostData, cos_d, context, stream);

    void* queryDeviceAddr = src0->data;
    void* keyDeviceAddr = dst->data;
    void* sinDeviceAddr = sinOutDeviceAddr;
    void* cosDeviceAddr = cosOutDeviceAddr;
    aclnn_shape_t queryShape = {1, size/ne[0], 1, ne[0]};
    aclnn_shape_t keyShape = queryShape;
    aclnn_shape_t sinShapeRp = {1, size/ne[0], 1, ne[0]};
    aclnn_shape_t cosShapeRp = sinShapeRp;

    ret = aclnn_rope_func(queryDeviceAddr, keyDeviceAddr, cosDeviceAddr, sinDeviceAddr,
                        queryShape, keyShape, cosShapeRp, sinShapeRp,
                        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ret = aclrtMemcpy(keyDeviceAddr, size * sizeof(float), queryDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return);
    // ret = aclnnRoPEFunc(queryShape, keyShape, cosShapeRp, sinShapeRp, queryHostData, keyHostData, cosQKHostData, sinQKHostData, dst, context, stream); 

    aclrtFree(powExpDeviceAddr);
    aclrtFree(powOutDeviceAddr);
    aclrtFree(mulOtherDeviceAddr);
    aclrtFree(mulSelfDeviceAddr);
    aclrtFree(mulOutDeviceAddr);
    aclrtFree(mulsOutDeviceAddr);
    aclrtFree(sinOutDeviceAddr);
    aclrtFree(cosOutDeviceAddr);
}