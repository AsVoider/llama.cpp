#include "aclnn-compute.h"
#include <iostream>
#include <vector>
#include <cstring>
#include "common.h"
#include "acl/acl.h"


void ggml_ascend_silu_test(int64_t lens, int64_t width, float* data, int32_t deviceId, aclrtStream stream){
    std::vector<int64_t> selfShape = {lens, width};
    std::vector<int64_t> otherShape = {lens, width};
    std::vector<int64_t> outShape = {lens, width};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData(data, data+ lens* width);
    std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> outHostData(lens* width, 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
    ctx->streams[deviceId][0] = stream;

    const int64_t ne[4] = {lens, width, 1, 1};

    ggml_tensor* src0 = new ggml_tensor();
    src0->ne[0] = ne[0];
    src0->ne[1] = ne[1];
    src0->ne[2] = ne[2];
    src0->ne[3] = ne[3];
    src0->type = GGML_TYPE_F32;
    src0->data = selfDeviceAddr;

    ggml_tensor* src1 = new ggml_tensor();
    src1->ne[0] = ne[0];
    src1->ne[1] = ne[1];
    src1->ne[2] = ne[2];
    src1->ne[3] = ne[3];
    src1->type = GGML_TYPE_F32;
    src1->data = otherDeviceAddr;

    ggml_tensor* dst = new ggml_tensor();
    dst->ne[0] = ne[0];
    dst->ne[1] = ne[1];
    dst->ne[2] = ne[2];
    dst->ne[3] = ne[3];
    dst->type = GGML_TYPE_F32;
    dst->data = outDeviceAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_ascend_silu(*ctx, dst);
    CHECK_RET(ret == ACL_SUCCESS, return);

    auto size = GetShapeSize(outShape);
     std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    return;
}

void ggml_ascend_cpy_test(int64_t* ne, float* data, int32_t deviceId, aclrtStream stream){

    std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
    std::vector<int64_t> otherShape = {ne[3], ne[2], ne[1], ne[0]};
    std::vector<int64_t> outShape = {ne[3], ne[2], ne[1], ne[0]};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData(data, data + ne[0]*ne[1]*ne[2]*ne[3]);
    std::vector<float> otherHostData(ne[0]*ne[1]*ne[2]*ne[3], 0);
    std::vector<float> outHostData(ne[0]*ne[1]*ne[2]*ne[3], 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
    ctx->streams[deviceId][0] = stream;

    ggml_tensor* src0 = new ggml_tensor();
    src0->ne[0] = ne[0];
    src0->ne[1] = ne[1];
    src0->ne[2] = ne[2];
    src0->ne[3] = ne[3];
    src0->type = GGML_TYPE_F32;
    src0->data = selfDeviceAddr;

    ggml_tensor* src1 = new ggml_tensor();
    src1->ne[0] = ne[0];
    src1->ne[1] = ne[1];
    src1->ne[2] = ne[2];
    src1->ne[3] = ne[3];
    src1->type = GGML_TYPE_F32;
    src1->data = otherDeviceAddr;

    ggml_tensor* dst = new ggml_tensor();
    dst->ne[0] = ne[0];
    dst->ne[1] = ne[1];
    dst->ne[2] = ne[2];
    dst->ne[3] = ne[3];
    dst->type = GGML_TYPE_F32;
    dst->data = outDeviceAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_ascend_cpy(*ctx, dst->src[0], dst);
    CHECK_RET(ret == ACL_SUCCESS, return);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void ggml_ascend_dup_test(int64_t* ne, float* data, int32_t deviceId, aclrtStream stream){

    std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
    std::vector<int64_t> otherShape = {ne[3], ne[2], ne[1], ne[0]};
    std::vector<int64_t> outShape = {ne[3], ne[2], ne[1], ne[0]};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData(data, data + ne[0]*ne[1]*ne[2]*ne[3]);
    std::vector<float> otherHostData(ne[0]*ne[1]*ne[2]*ne[3], 0);
    std::vector<float> outHostData(ne[0]*ne[1]*ne[2]*ne[3], 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
    ctx->streams[deviceId][0] = stream;

    ggml_tensor* src0 = new ggml_tensor();
    src0->ne[0] = ne[0];
    src0->ne[1] = ne[1];
    src0->ne[2] = ne[2];
    src0->ne[3] = ne[3];
    src0->type = GGML_TYPE_F32;
    src0->data = selfDeviceAddr;

    ggml_tensor* src1 = new ggml_tensor();
    src1->ne[0] = ne[0];
    src1->ne[1] = ne[1];
    src1->ne[2] = ne[2];
    src1->ne[3] = ne[3];
    src1->type = GGML_TYPE_F32;
    src1->data = otherDeviceAddr;

    ggml_tensor* dst = new ggml_tensor();
    dst->ne[0] = ne[0];
    dst->ne[1] = ne[1];
    dst->ne[2] = ne[2];
    dst->ne[3] = ne[3];
    dst->type = GGML_TYPE_F32;
    dst->data = outDeviceAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_ascend_dup(*ctx, dst);
    CHECK_RET(ret == ACL_SUCCESS, return);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void ggml_ascend_add_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream){

    std::vector<int64_t> selfShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
    std::vector<int64_t> otherShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
    std::vector<int64_t> outShape  = {
        (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
        (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
        (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
        (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
    };


    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData(data1, data1 + ne1[0]*ne1[1]*ne1[2]*ne1[3]);  
    std::vector<float> otherHostData(data2, data2 + ne2[0]*ne2[1]*ne2[2]*ne2[3]);
    std::vector<float> outHostData(outShape[0]*outShape[1]*outShape[2]*outShape[3], 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
    ctx->streams[deviceId][0] = stream;

    ggml_tensor* src0 = new ggml_tensor();
    src0->ne[0] = ne1[0];
    src0->ne[1] = ne1[1];
    src0->ne[2] = ne1[2];
    src0->ne[3] = ne1[3];
    src0->type = GGML_TYPE_F32;
    src0->data = selfDeviceAddr;

    ggml_tensor* src1 = new ggml_tensor();
    src1->ne[0] = ne2[0];
    src1->ne[1] = ne2[1];
    src1->ne[2] = ne2[2];
    src1->ne[3] = ne2[3];
    src1->type = GGML_TYPE_F32;
    src1->data = otherDeviceAddr;

    ggml_tensor* dst = new ggml_tensor();
    dst->ne[0] = ne1[0];
    dst->ne[1] = ne1[1];
    dst->ne[2] = ne1[2];
    dst->ne[3] = ne1[3];
    dst->type = GGML_TYPE_F32;
    dst->data = outDeviceAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_ascend_add(*ctx, dst);
    CHECK_RET(ret == ACL_SUCCESS, return);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void ggml_ascend_mul_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream){

    std::vector<int64_t> selfShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
    std::vector<int64_t> otherShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
    std::vector<int64_t> outShape  = {
        (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
        (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
        (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
        (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
    };
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData(data1, data1 + ne1[0]*ne1[1]*ne1[2]*ne1[3]);
    std::vector<float> otherHostData(data2, data2 + ne2[0]*ne2[1]*ne2[2]*ne2[3]);
    std::vector<float> outHostData(outShape[0]*outShape[1]*outShape[2]*outShape[3], 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
    ctx->streams[deviceId][0] = stream;

    ggml_tensor* src0 = new ggml_tensor();
    src0->ne[0] = ne1[0];
    src0->ne[1] = ne1[1];
    src0->ne[2] = ne1[2];
    src0->ne[3] = ne1[3];
    src0->type = GGML_TYPE_F32;
    src0->data = selfDeviceAddr;

    ggml_tensor* src1 = new ggml_tensor();
    src1->ne[0] = ne2[0];
    src1->ne[1] = ne2[1];
    src1->ne[2] = ne2[2];
    src1->ne[3] = ne2[3];
    src1->type = GGML_TYPE_F32;
    src1->data = otherDeviceAddr;

    ggml_tensor* dst = new ggml_tensor();
    dst->ne[0] = ne1[0];
    dst->ne[1] = ne1[1];
    dst->ne[2] = ne1[2];
    dst->ne[3] = ne1[3];
    dst->type = GGML_TYPE_F32;
    dst->data = outDeviceAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_ascend_mul(*ctx, dst);
    CHECK_RET(ret == ACL_SUCCESS, return);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void ggml_ascend_get_rows_test(int64_t*ne1, int64_t*ne2, float* data1, int64_t* data2, int32_t deviceId, aclrtStream stream){
    std::vector<int64_t> selfShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
    std::vector<int64_t> otherShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
    std::vector<int64_t> outShape = {1, 1, ne2[2]*ne2[1]*ne2[0], ne1[0]};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData(data1 ,data1+ ne1[0]*ne1[1]*ne1[2]*ne1[3]);
    std::vector<int64_t> otherHostData(data2, data2 + ne2[0]*ne2[1]*ne2[2]*ne2[3]);
    std::vector<float> outHostData(outShape[0]*outShape[1]*outShape[2]*outShape[3], 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);

    ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
    ctx->streams[deviceId][0] = stream;


    ggml_tensor* src0 = new ggml_tensor();
    src0->ne[0] = ne1[0];
    src0->ne[1] = ne1[1];
    src0->ne[2] = ne1[2];
    src0->ne[3] = ne1[3];
    src0->type = GGML_TYPE_F32;
    src0->data = selfDeviceAddr;

    ggml_tensor* src1 = new ggml_tensor();
    src1->ne[0] = ne2[0];
    src1->ne[1] = ne2[1];
    src1->ne[2] = ne2[2];
    src1->ne[3] = ne2[3];
    src1->type = GGML_TYPE_I64;
    src1->data = otherDeviceAddr;

    ggml_tensor* dst = new ggml_tensor();
    dst->ne[0] = ne1[0];
    dst->ne[1] =  ne2[2]*ne2[1]*ne2[0];
    dst->ne[2] = 1;
    dst->ne[3] = 1;
    dst->type = GGML_TYPE_F32;
    dst->data = outDeviceAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_ascend_get_rows(*ctx, dst);
    CHECK_RET(ret == ACL_SUCCESS, return);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void ggml_ascend_soft_max_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, float s, aclrtStream stream) {
    auto src0(new ggml_tensor());
    auto src1(new ggml_tensor());
    auto dst(new ggml_tensor());
    memcpy((void *)src0->ne, (void *)ne1, 4 * sizeof(int64_t));
    memcpy((void *)src1->ne, (void *)ne2, 4 * sizeof(int64_t));
    memcpy((void *)dst->ne, (void *)ne1, 4 * sizeof(int64_t));
    src0->type = GGML_TYPE_F32;
    src1->type = GGML_TYPE_F32;
    dst->type = GGML_TYPE_F32;

    void* dataAddr = nullptr;
    void* maskAddr = nullptr;
    void* outAddr = nullptr;
    aclnn_shape_t dataShape{ne1[3], ne1[2], ne1[1], ne1[0]};
    // for (auto &sp : dataShape) {
    //     printf("%ld ", sp);
    // }
    // printf("\n");
    std::vector<float> dataData(data1, data1 + ne1[0]*ne1[1]*ne1[2]*ne1[3]);
    auto ret = data_addr_malloc(dataShape, dataData, &dataAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc & cpy failed\n"); return);

    aclnn_shape_t maskShape{ne2[3], ne2[2], ne2[1], ne1[0]};
    std::vector<float> maskData(data2, data2 + ne2[0] * ne2[1] * ne2[2] * ne2[3]);
    ret = data_addr_malloc(maskShape, maskData, &maskAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc & cpy failed\n"); return);
    
    // for (auto &n : dataData) {
    //     printf("%f ", n);
    // }
    // printf("\n");
    // for (auto &n : maskData) {
    //     printf("%f ", n);
    // }
    // printf("\n");

    aclnn_shape_t outShape{ne1[3], ne1[2], ne1[1], ne1[0]};
    std::vector<float> outData(ne1[0]*ne1[1]*ne1[2]*ne1[3], 0);
    ret = data_addr_malloc(outShape, outData, &outAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc & cpy failed\n"); return);

    src0->data = dataAddr;
    src1->data = maskAddr;
    dst->data = outAddr;
    dst->src[0] = src0;
    dst->src[1] = src1;
    memcpy((void *)&dst->op_params[0], (void *)&s, sizeof(float));

    auto ctx(new ggml_backend_ascend_context(0));

    ggml_ascend_soft_max_new(*ctx, dst);
    auto size(GetShapeSize(outShape));
    std::vector<float> res(size, 0);
    ret = aclrtMemcpy(res.data(), res.size() * sizeof(res[0]), dst->data, res.size() * sizeof(res[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    FILE * sm = fopen("./sm.txt", "a+");
    for (int64_t i = 0; i < size; i++) {
       fprintf(sm, "result[%ld] is: %f\n", i, res[i]);
    }
}