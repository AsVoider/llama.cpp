#include <iostream>
#include <vector>
#include "common.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_hardsigmoid.h"
#include "aclnn-unary.h"
#include "aclnn-compute.h"


void aclnn_silu_func_test(int64_t lens, int64_t width, float* data, int32_t deviceId, ggml_backend_ascend_context & ctx){
    GGML_UNUSED(deviceId);
    GGML_UNUSED(ctx);

    std::vector<int64_t> selfShape = {lens, width};
    std::vector<int64_t> otherShape = {lens, width};
    std::vector<int64_t> outShape = {lens, width};
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    // aclTensor* self = nullptr;
    // aclTensor* other = nullptr;
    // aclScalar* alpha = nullptr;
    // aclTensor* out = nullptr;
    std::vector<float> selfHostData(data, data+ lens* width);
    std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> outHostData(lens* width, 0);

    auto ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);
    ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return);


    // aclnn_silu_func(selfDeviceAddr, outDeviceAddr, selfShape, outShape, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, ctx);

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