//
// Created by 35763 on 2024/6/26.
//
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_matmul.h"
#include "common.h"
#include "aclnn-mul.h"

int aclnn_mul_func(void* selfDataAddr, void* otherDataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& otherShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType otherDataType, aclDataType outDataType,
  aclrtStream &stream) {

  auto ret = 0;

  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(otherShape, otherDataType, &otherDataAddr, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return 0;
}

int aclnnMulFunc( std::vector<int64_t>& selfShape,
  std::vector<int64_t>& otherShape,
  std::vector<int64_t>& outShape,
  std::vector<float>& selfHostData,
  std::vector<float>& otherHostData,
  std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream){

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMul第一段接口
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMul第二段接口
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
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
  std::copy(resultData.data(), resultData.data() + resultData.size(), dst);
  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}


int aclnn_muls_func(void* selfDataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType outDataType,
  float otherValue, aclrtStream &stream) {

  auto ret = 0;

  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;

  ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnMulsGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMuls failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return 0;
}

int aclnnMulsFunc(std::vector<float>& selfHostData, std::vector<float>& outHostData, float otherValue, std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape ,float* dst, aclrtContext &context, aclrtStream &stream){


  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  // 创建self aclTensor
  int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclScalar
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMuls第一段接口
  ret = aclnnMulsGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMuls第二段接口
  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMuls failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
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

  std::copy(resultData.data(), resultData.data() + resultData.size(), dst);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int aclnn_mul_mat_func(void* selfDataAddr, void* mat2DataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& mat2Shape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType mat2DataType, aclDataType outDataType,
  aclrtStream &stream) {
  
  auto ret = 0;

  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;

  ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(mat2Shape, mat2DataType, &mat2DataAddr, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return 0;
}

int aclnnMulMatFunc(std::vector<float>& selfHostData, std::vector<float>& mat2HostData, std::vector<int64_t>& selfShape, std::vector<int64_t>& mat2Shape,std::vector<int64_t>& outShape, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream){

  // 2. 构造输入与输出，需要根据API的接口自定义构造

  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;

  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMatmul第一段接口
  ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMatmul第二段接口
  ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);

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
  std::copy(resultData.data(), resultData.data() + resultData.size(), dst);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);
  return 0;

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(mat2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

void aclnnMulTest(){
  std::vector<int64_t> selfShape = {2,2,1, 2};
  std::vector<int64_t> otherShape = {2,2,1, 2};
  std::vector<int64_t> outShape = {2,2,1, 2};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  float *a ;
  //int ret = aclnnMulFunc(selfShape, otherShape, outShape, selfHostData, otherHostData, outHostData,a);
}


void aclnnMulsTest(){
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  float alphaValue = 1.2f;
  int len = 2;
  int width = 4;
  float dstTestp[len*width];
  std::vector<int64_t> selfShape = {len, width};
  std::vector<int64_t> outShape = {len, width};
  std::vector<float> outHostData(len* width, 0);
  //int res = aclnnMulsFunc(selfHostData, alphaValue, selfShape, outShape, outHostData, dstTestp);
}

void aclnnMulMatTest(){
  std::vector<int64_t> selfShape1 = {2,2,2,2,8,8};
  std::vector<int64_t> mat2Shape1 = {8,16};
  std::vector<int64_t> outShape1 = {2,2,2,2,8,16};
  std::vector<float> selfHostData1(1024, 1);
  std::vector<float> mat2HostData1(128, 1);
  std::vector<float> outHostData1(2048, 0);
 // int ret = aclnnMulMatFunc(selfHostData1, mat2HostData1, selfShape1, mat2Shape1, outShape1, outHostData1);
}


