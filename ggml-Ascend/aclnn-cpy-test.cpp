#include "aclnn-cpy.h"
#include <iostream>
#include <vector>
#include "common.h"
#include "acl/acl.h"


void aclnn_cpy_func_test(int64_t* ne, float* data, int32_t deviceId, aclrtStream stream){

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

    ret = aclnn_cpy_func(outDeviceAddr, selfDeviceAddr, outShape, selfShape, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, stream);
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

}

