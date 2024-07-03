#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnnSort.h"
#include "aclnnop/aclnn_argsort.h"

int aclnnArgSortFunc(int64_t dim ,bool descending, std::vector<int64_t>& selfShape, std::vector<int64_t>& outIndicesShape, std::vector<int64_t>& selfHostData, std::vector<int64_t>& outIndicesHostData){
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    void* selfDeviceAddr = nullptr;
    void* outIndicesDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* outIndices = nullptr;
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outValues和outIndices aclTensor
    ret = CreateAclTensor(outIndicesHostData, outIndicesShape, &outIndicesDeviceAddr, aclDataType::ACL_INT64, &outIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnArgsort第一段接口
    ret = aclnnArgsortGetWorkspaceSize(self, dim, descending, outIndices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgsortGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnArgsort第二段接口
    ret = aclnnArgsort(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgsort failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size2 = GetShapeSize(outIndicesShape);
    std::vector<int64_t> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), outIndicesDeviceAddr,
                      size2 * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("result indices [%ld] is: %ld\n", i, resultData2[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(outIndices);
    
     // 7. 释放device 资源
    aclrtFree(selfDeviceAddr);
    aclrtFree(outIndicesDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
   
    return 0;

}


void aclnnArgSortTest(){
    int64_t dim = 0;
    bool descending = false;
    std::vector<int64_t> selfShape = {3, 4};
    std::vector<int64_t> outIndicesShape = {3, 4};
    std::vector<int64_t> selfHostData = {7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};
    std::vector<int64_t> outIndicesHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int ret = aclnnArgSortFunc(dim, descending, selfShape, outIndicesShape, selfHostData, outIndicesHostData);
}