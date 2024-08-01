#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_im2col.h"
#include "aclnn-im2col.h"
#include "common.h"

int aclnnIm2ColFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<int64_t>& kernelSizeData, std::vector<int64_t>& dilationData, std::vector<int64_t>& paddingData, std::vector<int64_t>& strideData, std::vector<float>& outHostData){
  // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* kernelSize = nullptr;
  aclIntArray* dilation = nullptr;
  aclIntArray* padding = nullptr;
  aclIntArray* stride = nullptr;
  aclTensor* out = nullptr;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建aclIntArray
  kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  CHECK_RET(kernelSize != nullptr, return ret);
  dilation = aclCreateIntArray(dilationData.data(), 2);
  CHECK_RET(dilation != nullptr, return ret);
  padding = aclCreateIntArray(paddingData.data(), 2);
  CHECK_RET(padding != nullptr, return ret);
  stride = aclCreateIntArray(strideData.data(), 2);
  CHECK_RET(stride != nullptr, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnIm2col第一段接口
  ret = aclnnIm2colGetWorkspaceSize(self, kernelSize, dilation, padding, stride, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2colGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnIm2col第二段接口
  ret = aclnnIm2col(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2col failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(dilation);
  aclDestroyIntArray(padding);
  aclDestroyIntArray(stride);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

void aclnnIm2ColTest(){
  std::vector<int64_t> selfShape = {2, 2, 3};
  std::vector<int64_t> outShape = {8, 4};

  std::vector<float> selfHostData = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> dilationData = {1, 1};
  std::vector<int64_t> paddingData = {1, 1};
  std::vector<int64_t> strideData = {2, 2};
  std::vector<float> outHostData = {0.0};

  int ret = aclnnIm2ColFunc(selfShape, outShape, selfHostData, kernelSizeData, dilationData, paddingData, strideData, outHostData);
  GGML_UNUSED(ret);
}