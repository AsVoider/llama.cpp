#include "aclnn-norm.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnnop/aclnn_norm.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "aclnnop/aclnn_group_norm.h"


int aclnnNormFunc(std::vector<float>& selfHostData, std::vector<float>& outHostData, std::vector<int64_t>& dimData, float pValue, bool keepDim, std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape){

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
  aclTensor* out = nullptr;
  aclScalar* pScalar = nullptr;
  aclIntArray* dim = nullptr;


  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建p aclTensor
  pScalar = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
  CHECK_RET(pScalar != nullptr, return ret);
  // 创建dim aclIntArray
  dim = aclCreateIntArray(dimData.data(), 1);
  CHECK_RET(dim != nullptr, return ret);


  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnNorm第一段接口
  ret = aclnnNormGetWorkspaceSize(self, pScalar, dim, keepDim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnNorm第二段接口
  ret = aclnnNorm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNorm failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(dim);
  aclDestroyScalar(pScalar);
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


int aclnnGroupNormFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& gammaShape, std::vector<int64_t>& betaShape, std::vector<int64_t>& outShape,  std::vector<int64_t>& meanOutShape, 
        std::vector<int64_t>& rstdOutShape, std::vector<float>& selfHostData, std::vector<float>& gammaHostData, std::vector<float>& betaHostData, std::vector<float>& outHostData, 
        std::vector<float>& meanOutHostData, std::vector<float>& rstdOutHostData, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps){

            // 1. （固定写法）device/context/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造

  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建beta aclTensor
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建meanOut aclTensor
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rstdOut aclTensor
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupNorm第一段接口
  ret = aclnnGroupNormGetWorkspaceSize(self, gamma, beta, N, C, HxW, group, eps, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupNorm第二段接口
  ret = aclnnGroupNorm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNorm failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %f\n", i, outResultData[i]);
  }

  size = GetShapeSize(meanOutShape);
  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
  }

  size = GetShapeSize(rstdOutShape);
  std::vector<float> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(), rstdResultData.size() * sizeof(rstdResultData[0]), rstdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, rstdResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  aclrtFree(rstdOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;



}

int aclnn_rms_norm_func(void* xDataAddr, void* gammaDataAddr, void* yDataAddr, void* rstdDataAddr,
  aclnn_shape_t& xShape, aclnn_shape_t& gammaShape, aclnn_shape_t& yShape, aclnn_shape_t& rstdShape,
  aclDataType xDataType, aclDataType gammaDataType, aclDataType yDataType, aclDataType rstdDataType,
  float epsilon, aclrtStream &stream) {
  
  auto ret = 0;

  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* y = nullptr;
  aclTensor* rstd = nullptr;

  ret = create_acl_tensor(xShape, xDataType, &xDataAddr, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(gammaShape, gammaDataType, &gammaDataAddr, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(yShape, yDataType, &yDataAddr, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(rstdShape, rstdDataType, &rstdDataAddr, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnRmsNormGetWorkspaceSize(x, gamma, epsilon, y, rstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(y);
  aclDestroyTensor(rstd);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return 0;
}

int aclnnRmsNormFunc( std::vector<int64_t>& xShape,
  std::vector<int64_t>& gammaShape,
  std::vector<int64_t>& yShape,
  std::vector<int64_t>& rstdShape,
  std::vector<float>& xHostData,
  std::vector<float>& gammaHostData,
  std::vector<float>& yHostData,
  std::vector<float>& rstdHostData,
  float epsilon, float* dst, aclrtContext &context, aclrtStream &stream){

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* y = nullptr;
  aclTensor* rstd = nullptr;


  // 创建x aclTensor
  int ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rstd aclTensor
  ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的HostApi
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRmsNorm第一段接口
  ret = aclnnRmsNormGetWorkspaceSize(x, gamma, epsilon, y, rstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnRmsNorm第二段接口
  ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  std::copy(resultData.data(), resultData.data() + resultData.size(), dst);
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(y);
  aclDestroyTensor(rstd);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(rstdDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}






void aclnnNormTest(){
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {1, 2};
    std::vector<float> selfHostData = {0.0, 1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1};
    std::vector<float> outHostData = {0.0, 0.0};
    std::vector<int64_t> dimData = {0};
    float pValue = 2.0f;
    bool keepDim = true;
    int ret = aclnnNormFunc(selfHostData, outHostData, dimData, pValue, keepDim, selfShape, outShape);

}

void aclnnGroupNormTest(){
    std::vector<int64_t> selfShape = {2, 3, 4};
    std::vector<int64_t> gammaShape = {3};
    std::vector<int64_t> betaShape = {3};
    std::vector<int64_t> outShape = {2, 3, 4};
    std::vector<int64_t> meanOutShape = {2, 1};
    std::vector<int64_t> rstdOutShape = {2, 1};
    std::vector<float> selfHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> gammaHostData = {2.0, 2, 2};
    std::vector<float> betaHostData = {2.0, 2, 2};
    std::vector<float> outHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> meanOutHostData = {2.0, 2};
    std::vector<float> rstdOutHostData = {2.0, 2};
    int64_t N = 2;
    int64_t C = 3;
    int64_t HxW = 4;
    int64_t group = 1;
    double eps = 1e-5;
    int ret = aclnnGroupNormFunc(selfShape, gammaShape, betaShape, outShape, meanOutShape, rstdOutShape, selfHostData, gammaHostData, betaHostData, outHostData, meanOutHostData, rstdOutHostData,
        N, C, HxW, group, eps);

}

void aclnnRmsNormTest(){
  std::vector<int64_t> xShape = {1,1, 2, 16};
  std::vector<int64_t> gammaShape = {16};
  std::vector<int64_t> yShape = {1,1, 2, 16};
  std::vector<int64_t> rstdShape = {16, 1};
  std::vector<float> xHostData = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> gammaHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> yHostData(xHostData.size(), 0);
  std::vector<float> rstdHostData = {1, 16};
  float epsilon = 1;
  //int ret = aclnnRmsNormFunc(xShape, gammaShape, yShape, rstdShape, xHostData, gammaHostData, yHostData, rstdHostData, epsilon);
}
