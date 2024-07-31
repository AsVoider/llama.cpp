#include "aclnn-compute.h"
#include "aclnn-comp.h"
#include "string.h"
#include "aclnn-add.h"
#include "aclnn-mul.h"
#include "aclnn-cpy.h"
#include "aclnn-unary.h"
#include "aclnn-comp.h"
#include "aclnn-norm.h"
#include "aclnn-math.h"
#include "aclnn-rope.h"
#include "aclnn-permute.h"
#include "aclnn-repeat.h"
#include <cstring>
#include <cmath>
#include <algorithm>

void ggml_ascend_get_rows(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    // FILE * f = fopen("get_rows_npu_bamboo.txt", "a");
    // fprintf(f, "get_rows_npu\n");
    // fprintf(f, "src0 type: %d\n", src0->type);
    // fprintf(f, "src1 type: %d\n", src1->type);
    // fprintf(f, "dst type: %d\n", dst->type);
    // fprintf(f, "src0 ne: %ld %ld %ld %ld\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    // fprintf(f, "src1 ne: %ld %ld %ld %ld\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    // fprintf(f, "dst ne: %ld %ld %ld %ld\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    // fclose(f);

    aclnn_shape_t selfShape = {1, 1, ne01 * ne02 * ne03, ne00};
    aclnn_shape_t indexShape = {ne10 * ne11 * ne12};
    aclnn_shape_t outShape = {1, 1, ne10 * ne11 * ne12, ne00};
//     std::vector<int64_t> tmpHostData(ne10 * ne11 * ne12 * ne13, 0);
//     std::vector<int64_t> offset;
//     for(int i = 0; i < ne11 * ne12 * ne13; i++) {
//         offset.insert(offset.end(), ne10, i * ne01);
//     }
// 
//     void* offsetDeviceAddr = nullptr;
//     void* tmpDeviceAddr = nullptr;
//     
//     int ret = data_addr_malloc(indexShape, offset, &offsetDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);
// 
//     ret = data_addr_malloc(indexShape, tmpHostData, &tmpDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);
// 
//     ret = aclnn_add_func(src1->data, offsetDeviceAddr, tmpDeviceAddr,
//                         indexShape, indexShape, indexShape,
//                         ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type],
//                         stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);

    auto ret = aclnn_get_rows_func(src0->data, src1->data, dst->data,
                            selfShape, indexShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);

    // aclrtFree(offsetDeviceAddr);
    // aclrtFree(tmpDeviceAddr);
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_add failed. ERROR: %d\n", ret); return);
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul failed. ERROR: %d\n", ret); return);
}

void ggml_ascend_cpy(ggml_backend_ascend_context &ctx, ggml_tensor *src, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    aclnn_shape_t srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    aclnn_shape_t dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    srcShape = dstShape;

    // std::cout<<"in ggml_ascend_cpy :"<<std::endl;
    // std::cout<<"src shape: "<<src->ne[3]<<" "<<src->ne[2]<<" "<<src->ne[1]<<" "<<src->ne[0]<<std::endl;
    // std::cout<<"dst shape: "<<dst->ne[3]<<" "<<dst->ne[2]<<" "<<dst->ne[1]<<" "<<dst->ne[0]<<std::endl;
    // std::cout<<"src name: "<<src->name<<std::endl;
    // std::cout<<"dst name: "<<dst->name<<std::endl;


    // int ret = aclnn_cpy_func(dst->data, src->data,
    //                         dstShape, srcShape,
    //                         ggml_to_acl_map[dst->type], ggml_to_acl_map[src->type],
    //                         stream);

    int ret = aclnn_cpy_func(dst->data, src->data,
                            dstShape, srcShape,
                            ggml_to_acl_map[dst->type], ggml_to_acl_map[src->type],
                            stream,
                            dst->nb, src->nb);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_cpy failed. ERROR: %d\n", ret); return);
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_silu failed. ERROR: %d\n", ret); return);
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rms_norm failed. ERROR: %d\n", ret); return);
    ret = data_addr_malloc(rstdShape, rstdHostData, &rstdDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rms_norm failed. ERROR: %d\n", ret); return);

    float eps;
    memcpy(&eps, (float*)dst->op_params+0, sizeof(float));

    ret = aclnn_rms_norm_func(src0->data, gammaDeviceAddr, dst->data, rstdDeviceAddr,
                                xShape, gammaShape, yShape, rstdShape,
                                ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                eps, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rms_norm failed. ERROR: %d\n", ret); return);

    aclrtFree(gammaDeviceAddr);
    aclrtFree(rstdDeviceAddr);
}

void ggml_ascend_soft_max_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    auto src0 = dst->src[0], src1 = dst->src[1];
    auto stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    auto dataAddr = src0->data;
    auto maskAddr = src1->data;
    auto outAddr = dst->data;

    float scale;
    memcpy((void *)&scale, (void *)&dst->op_params[0], sizeof(float));

    aclnn_shape_t dataShape{ne03, ne02, ne01, ne00};
    aclnn_shape_t maskShape{ne12, ne01, ne10};
    aclnn_shape_t outShape{ne3, ne2, ne1, ne0};

    auto ret = aclnn_soft_max_func(dataAddr, maskAddr, scale, outAddr, dataShape, maskShape, outShape, 
    ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type], stream);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s failed. ERROR: %d\n", __func__, ret); return);
}

void ggml_ascend_soft_max(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0]; // 
    ggml_tensor* src1 = dst->src[1]; // mask
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    void* mulsSelfDeviceAddr = src0->data;
    void* mulsOutDeviceAddr = nullptr;
    // float scale = dst->op_params[0];
    float scale;
    memcpy((void *)&scale, (void *)&dst->op_params[0], sizeof(float));
    aclnn_shape_t mulsSelfShape = {ne00, ne01, ne02, ne03};
    aclnn_shape_t mulsOutShape = mulsSelfShape;
    std::vector<float> mulsOutHostData(ne00 * ne01 * ne02 * ne03, 0);
    auto ret = data_addr_malloc(mulsOutShape, mulsOutHostData, &mulsOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    ret = aclnn_muls_func(mulsSelfDeviceAddr, mulsOutDeviceAddr, mulsSelfShape, mulsOutShape, ACL_FLOAT, ACL_FLOAT, scale, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\nmulsOut: \n");
    // auto tmp_size = GetShapeSize(mulsOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), mulsOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* addSelfDeviceAddr = mulsOutDeviceAddr;
    void* addOtherDeviceAddr = src1->data;
    void* addOutDeviceAddr = nullptr;
    aclnn_shape_t addSelfShape = mulsOutShape;
    aclnn_shape_t addOtherShape = {ne10, ne01, ne12, ne13};
    aclnn_shape_t addOutShape = {
        (addSelfShape[0] > addOtherShape[0]) ? addSelfShape[0] : addOtherShape[0],
        (addSelfShape[1] > addOtherShape[1]) ? addSelfShape[1] : addOtherShape[1],
        (addSelfShape[2] > addOtherShape[2]) ? addSelfShape[2] : addOtherShape[2],
        (addSelfShape[3] > addOtherShape[3]) ? addSelfShape[3] : addOtherShape[3]
    };
    std::vector<float> addOutHostData(addOutShape[0] * addOutShape[1] * addOutShape[2] * addOutShape[3], 0);
    ret = data_addr_malloc(addOutShape, addOutHostData, &addOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    ret = aclnn_add_func(addSelfDeviceAddr, addOtherDeviceAddr, addOutDeviceAddr,
                        addSelfShape, addOtherShape, addOutShape,
                        ACL_FLOAT, ggml_to_acl_map[src1->type], ACL_FLOAT,
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\naddOut: \n");
    // auto tmp_size = GetShapeSize(addOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), addOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* softMaxSelfDeviceAddr = addOutDeviceAddr;
    void* softMaxOutDeviceAddr = dst->data;
    aclnn_shape_t softMaxSelfShape = addOutShape;
    aclnn_shape_t softMaxOutShape = softMaxSelfShape;
    ret = aclnn_soft_max_func(softMaxSelfDeviceAddr, softMaxOutDeviceAddr,
                            softMaxSelfShape, softMaxOutShape,
                            ACL_FLOAT, ACL_FLOAT,
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    aclrtFree(mulsOutDeviceAddr);
    aclrtFree(addOutDeviceAddr);
}

void ggml_ascend_mul_mat(ggml_backend_ascend_context &ctx, ggml_tensor *src0, ggml_tensor *src1, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    // std::cout<<"in ggml_ascend_mat_mul :"<<std::endl;
    // std::cout<<"src0 shape: "<<src0->ne[3]<<" "<<src0->ne[2]<<" "<<src0->ne[1]<<" "<<src0->ne[0]<<std::endl;
    // std::cout<<"src1 shape: "<<src1->ne[3]<<" "<<src1->ne[2]<<" "<<src1->ne[1]<<" "<<src1->ne[0]<<std::endl;
    // std::cout<<"dst shape: "<<dst->ne[3]<<" "<<dst->ne[2]<<" "<<dst->ne[1]<<" "<<dst->ne[0]<<std::endl;
    // std::cout<<"src name: "<<src0->name<<" "<<src1->name<<std::endl;
    // std::cout<<"dst name: "<<dst->name<<std::endl;
    // std::cout<<"src0 type: "<< src0->type << std::endl;
    // std::cout<<"src1 type: "<< src1->type << std::endl;
    // std::cout<<"dst type: "<< dst->type << std::endl;


    CHECK_RET((ne03 == ne13 && ne13 == 1), LOG_PRINT("error: ne03: %ld, ne13: %ld\n", ne03, ne13));
    CHECK_RET((ne12 >= ne02 && ne12 % ne02 == 0), LOG_PRINT("error: ne12: %ld, ne02: %ld\n", ne12, ne02));

    void* permuteSelfDeviceAddr = src0->data;
    void* permuteOutDeviceAddr = nullptr;
    aclnn_shape_t permuteSelfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t permuteOutShape = {ne03, ne02, ne00, ne01};
    aclnn_shape_t permuteDims = {0, 1, 3, 2};
    int ret = addr_malloc(permuteOutShape, &permuteOutDeviceAddr, ggml_type_size_t[src0->type]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    ret = aclnn_permute_func(permuteSelfDeviceAddr, permuteOutDeviceAddr,
                                permuteSelfShape, permuteOutShape,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src0->type],
                                permuteDims, stream);

    // LOG_PRINT("\naclnn_permute_func: \n");
    // auto tmp_size = GetShapeSize(permuteOutShape);
    // std::vector<aclFloat16> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), permuteOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    // }

    void* cpySelfRefDeviceAddr = nullptr;
    void* cpySrcDeviceAddr = permuteOutDeviceAddr;
    aclnn_shape_t cpySelfRefShape = {ne03, ne02, ne00, ne01};
    aclnn_shape_t cpySrcShape = permuteOutShape;
    ret = addr_malloc(cpySelfRefShape, &cpySelfRefDeviceAddr, ggml_type_size_t[src1->type]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    ret = aclnn_cpy_func(cpySelfRefDeviceAddr, cpySrcDeviceAddr,
                        cpySelfRefShape, cpySrcShape,
                        ggml_to_acl_map[src1->type], ggml_to_acl_map[src0->type],
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    void* boardcastSrc0DeviceAddr = nullptr;
    int64_t size = ne03 * ne12 * ne01 * ne00;
    ret = aclrtMalloc(&boardcastSrc0DeviceAddr, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);
    for (int i = 0; i < size; i += ne02 * ne01 * ne00) {
        ret = aclrtMemcpy((void *)((float *)boardcastSrc0DeviceAddr + i), ne02 * ne01 * ne00 * sizeof(float), cpySelfRefDeviceAddr, ne02 * ne01 * ne00 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    }

    // LOG_PRINT("\boardcastSrc0DeviceAddr: \n");
    // auto tmp_size = size;
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), boardcastSrc0DeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    // LOG_PRINT("\aclnn_cpy_func: \n");
    // auto tmp_size = GetShapeSize(cpySelfRefShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cpySelfRefDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* batchMatMulSelfDeviceAddr = src1->data;
    void* batchMatMulMat2DeviceAddr = boardcastSrc0DeviceAddr;
    void* batchMatMulOutDeviceAddr = dst->data;
    aclnn_shape_t batchMatMulSelfShape = {ne12, ne11, ne10};
    aclnn_shape_t batchMatMulMat2Shape = {ne12, ne00, ne01};
    aclnn_shape_t batchMatMulOutShape = {ne2, ne1, ne0};

    // LOG_PRINT("ne12: %ld, ne11: %ld, ne10: %ld\n", ne12, ne11, ne10);
    // LOG_PRINT("ne02: %ld, ne00: %ld, ne01: %ld\n", ne02, ne00, ne01);
    // LOG_PRINT("ne2: %ld, ne1: %ld, ne0: %ld\n", ne2, ne1, ne0);

    // LOG_PRINT("\abatchMatMulSelfDeviceAddr: \n");
    // auto tmp_size = GetShapeSize(batchMatMulSelfShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), batchMatMulSelfDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    ret = aclnn_batch_mat_mul_func(batchMatMulSelfDeviceAddr, batchMatMulMat2DeviceAddr, batchMatMulOutDeviceAddr,
                                batchMatMulSelfShape, batchMatMulMat2Shape, batchMatMulOutShape,
                                ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                                stream);

    aclrtFree(permuteOutDeviceAddr);
    aclrtFree(cpySelfRefDeviceAddr);
    aclrtFree(boardcastSrc0DeviceAddr);
}

void ggml_ascend_rope(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {

    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    int64_t* ne = src0->ne;
    float freq_scale;
    float freq_base;
    int n_dims = ((int32_t *) dst->op_params)[1];
    void* pos = src1->data;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));

    int64_t size = ne[0] * ne[1] * ne[2] * ne[3];


    // std::cout<<"in Rope"<<std::endl;
    // std::cout<<ne[0]<<" "<<ne[1]<<" "<<ne[2]<<" "<<ne[3]<<std::endl;

    // float theta_scale_pow[ne[0] / 2];
    // float theta_base[size];
    // float theta[size];
    // float sin_d[size];
    // float cos_d[size];

    float theta_scale = pow(freq_base, -2.0 / n_dims);
    // LOG_PRINT("theta_scale: %f\n", theta_scale);

    std::vector<float> powExpHostData(ne[0]/2, 0);
    std::vector<float> powOutData = powExpHostData;
    aclnn_shape_t powExpShape = {ne[0]/2, 1};
    aclnn_shape_t powOutShape = powExpShape;
    for(decltype(powExpHostData.size()) i(0); i < powExpHostData.size(); i++){
        powExpHostData[i] = i + 1;
    }
    void* powExpDeviceAddr = nullptr;
    void* powOutDeviceAddr = nullptr;
    int ret = data_addr_malloc(powExpShape, powExpHostData, &powExpDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    ret = data_addr_malloc(powOutShape, powOutData, &powOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    ret = aclnn_pow_scalar_tensor_func(theta_scale, powExpDeviceAddr, powOutDeviceAddr,
        powExpShape, powOutShape,
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\npowOut: \n");
    // auto tmp_size = GetShapeSize(powOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), powOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* mulOtherDeviceAddr = nullptr;
    ret = aclrtMalloc(&mulOtherDeviceAddr, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // for (int i = 0; i < size; i += ne[0] / 2) {
    //     ret = aclrtMemcpy((void *)((float *)mulOtherDeviceAddr + i), ne[0] / 2 * sizeof(float), powOutDeviceAddr, ne[0] / 2 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // }
    aclnn_shape_t repeatSelfShape = {1, 1, ne[0]/2};
    aclnn_shape_t repeatOutShape = {ne[2], ne[1], ne[0]};
    std::vector<int64_t> repeatsArray = {ne[2], ne[1], 2};
    ret = aclnn_repeat_func(powOutDeviceAddr, mulOtherDeviceAddr,
                            repeatSelfShape, repeatOutShape,
                            ACL_FLOAT, ACL_FLOAT,
                            repeatsArray, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    ret = aclrtMalloc(&mulSelfDeviceAddr, size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // for (int i = 0; i < size; i += ne[2]) {
    //     ret = aclrtMemcpy((void *)((int32_t *)mulSelfDeviceAddr + i), ne[2] * sizeof(int32_t), pos, ne[2] * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // }
    aclnn_shape_t repeatSelfShape2 = {ne[2], 1, 1};
    aclnn_shape_t repeatOutShape2 = {ne[2], ne[1], ne[0]};
    std::vector<int64_t> repeatsArray2 = {1, ne[1], ne[0]};
    ret = aclnn_repeat_func(pos, mulSelfDeviceAddr,
                            repeatSelfShape2, repeatOutShape2,
                            ACL_INT32, ACL_INT32,
                            repeatsArray2, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\nmulOut: \n");
    // auto tmp_size = GetShapeSize(mulOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), mulOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    // ret = aclnnMulFunc(mulSelfShape, mulOtherShape, mulOutShape, mulSelfHostData, mulOtherHostData, mulOutHostData, theta_base, context, stream);

    void* mulsOutDeviceAddr = nullptr;
    void* mulsSelfDeviceAddr = mulOutDeviceAddr;
    aclnn_shape_t mulsSelfShape = mulSelfShape;
    aclnn_shape_t mulsOutShape = mulOutShape;
    std::vector<float> mulsOutHostData(size, 0);
    ret = data_addr_malloc(mulsOutShape, mulsOutHostData, &mulsOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // std::vector<float> mulsSelfHostData(theta_base, theta_base+ size);

    // for (int i=0; i< mulsSelfHostData.size(); i++){
    //   std::cout << mulsSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_muls_func(mulsSelfDeviceAddr, mulsOutDeviceAddr, mulsSelfShape, mulsOutShape, ACL_FLOAT, ACL_FLOAT, freq_scale, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnMulsFunc(mulsSelfHostData, mulsOutHostData, freq_scale, mulsSelfShape, mulsOutShape, theta, context, stream);

    // LOG_PRINT("\nmulsOut: \n");
    // auto tmp_size = GetShapeSize(mulsOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), mulsOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* sinSelfDeviceAddr = mulsOutDeviceAddr;
    void* sinOutDeviceAddr = nullptr;
    aclnn_shape_t sinSelfShape = {size, 1};
    aclnn_shape_t sinOutShape = {size, 1};
    std::vector<float> sinOutHostData(size, 0);
    ret = data_addr_malloc(sinOutShape, sinOutHostData, &sinOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // std::vector<float> sinSelfHostData(theta, theta+ size);


    // for (int i=0; i< sinSelfHostData.size(); i++){
    //   std::cout << sinSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_sin_func(sinSelfDeviceAddr, sinOutDeviceAddr, sinSelfShape, sinOutShape, ACL_FLOAT, ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnSinFunc(sinSelfShape, sinShape, sinSelfHostData, sinHostData, sin_d, context, stream);

    // LOG_PRINT("\nsinOut: \n");
    // auto tmp_size = GetShapeSize(sinOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), sinOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* cosSelfDeviceAddr = mulsOutDeviceAddr;
    void* cosOutDeviceAddr = nullptr;
    aclnn_shape_t cosSelfShape = {size, 1};
    aclnn_shape_t cosOutShape = {size, 1};
    std::vector<float> cosOutHostData(size, 0);
    ret = data_addr_malloc(cosOutShape, cosOutHostData, &cosOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // std::vector<float> cosSelfHostData(theta, theta+ size);

    ret = aclnn_cos_func(cosSelfDeviceAddr, cosOutDeviceAddr, cosSelfShape, cosOutShape, ACL_FLOAT, ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnCosFunc(cosSelfShape, cosShape, cosSelfHostData, cosHostData, cos_d, context, stream);

    // LOG_PRINT("\ncosOut: \n");
    // auto tmp_size = GetShapeSize(cosOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cosOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* permuteSelfDeviceAddr = src0->data;
    void* permuteOutDeviceAddr = nullptr;
    aclnn_shape_t permuteSelfShape = {1, size/ne[0], ne[0]/2, 2};
    aclnn_shape_t permuteOutShape = {1, size/ne[0], 2, ne[0]/2};
    aclnn_shape_t permuteDims = {0, 1, 3, 2};
    ret = addr_malloc(permuteOutShape, &permuteOutDeviceAddr, ggml_type_size_t[src0->type]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    ret = aclnn_permute_func(permuteSelfDeviceAddr, permuteOutDeviceAddr,
                                permuteSelfShape, permuteOutShape,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src0->type],
                                permuteDims, stream);

    void* queryDeviceAddr = permuteOutDeviceAddr;
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // ret = aclrtMemcpy(keyDeviceAddr, size * sizeof(float), queryDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnRoPEFunc(queryShape, keyShape, cosShapeRp, sinShapeRp, queryHostData, keyHostData, cosQKHostData, sinQKHostData, dst, context, stream); 

    void* permuteSelfDeviceAddr1 = queryDeviceAddr;
    void* permuteOutDeviceAddr1 = keyDeviceAddr;
    aclnn_shape_t permuteSelfShape1 = {1, size/ne[0], 2, ne[0]/2};
    aclnn_shape_t permuteOutShape1 = {1, size/ne[0], ne[0]/2, 2};
    aclnn_shape_t permuteDims1 = {0, 1, 3, 2};

    ret = aclnn_permute_func(permuteSelfDeviceAddr1, permuteOutDeviceAddr1,
                                permuteSelfShape1, permuteOutShape1,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src0->type],
                                permuteDims1, stream);

    aclrtFree(powExpDeviceAddr);
    aclrtFree(powOutDeviceAddr);
    aclrtFree(mulOtherDeviceAddr);
    aclrtFree(mulSelfDeviceAddr);
    aclrtFree(mulOutDeviceAddr);
    aclrtFree(mulsOutDeviceAddr);
    aclrtFree(sinOutDeviceAddr);
    aclrtFree(cosOutDeviceAddr);
    aclrtFree(permuteOutDeviceAddr);
}