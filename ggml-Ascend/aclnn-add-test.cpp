#include "aclnn-add.h"
#include <iostream>
#include <vector>
#include "common.h"
#include "acl/acl.h"

void aclnn_add_func_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream){

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

    ret = aclnn_add_func(selfDeviceAddr, otherDeviceAddr, outDeviceAddr, selfShape, otherShape, outShape, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, stream);
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}