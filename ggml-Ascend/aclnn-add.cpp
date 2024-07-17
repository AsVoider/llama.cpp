//
// Created by 35763 on 2024/6/26.
//
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnn-add.h"
#include "common.h"

int aclnn_add_func(void* selfDataAddr, void* otherDataAddr, void* outDataAddr,
	  aclnn_shape_t& selfShape, aclnn_shape_t& otherShape, aclnn_shape_t& outShape,
    aclDataType selfDataType, aclDataType otherDataType, aclDataType outDataType,
    aclrtStream &stream) {
    
    auto ret = 0;

    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = create_acl_tensor(otherShape, otherDataType, &otherDataAddr, &other);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    aclFloat16 alphaValue_f16 = aclFloatToFloat16(1.0f);
    float alphaValue_f32 = 1.0f;
    double alphaValue_f64 = 1.0;
    int8_t alphaValue_i8 = 1;
    int16_t alphaValue_i16 = 1;
    int32_t alphaValue_i32 = 1;
    int64_t alphaValue_i64 = 1;

    switch(outDataType) {
      case ACL_FLOAT16:
        alpha = aclCreateScalar(&alphaValue_f16, aclDataType::ACL_FLOAT16);
        break;
      case ACL_FLOAT:
        alpha = aclCreateScalar(&alphaValue_f32, aclDataType::ACL_FLOAT);
        break;
      case ACL_DOUBLE:
        alpha = aclCreateScalar(&alphaValue_f64, aclDataType::ACL_DOUBLE);
        break;
      case ACL_INT8:
        alpha = aclCreateScalar(&alphaValue_i8, aclDataType::ACL_INT8);
        break;
      case ACL_INT16:
        alpha = aclCreateScalar(&alphaValue_i16, aclDataType::ACL_INT16);
        break;
      case ACL_INT32:
        alpha = aclCreateScalar(&alphaValue_i32, aclDataType::ACL_INT32);
        break;
      case ACL_INT64:
        alpha = aclCreateScalar(&alphaValue_i64, aclDataType::ACL_INT64);
        break;
      default:
        LOG_PRINT("Unsupported data type\n");
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);
    
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

int aclnnAddFunc(std::vector<float>& selfHostData,std::vector<float>& otherHostData, float alphaValue, std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,float* dst, aclrtStream &stream){

  size_t length = selfHostData.size();
  std::vector<float> outHostData(length, 0);
  std::vector<float> errorRet(1, 0);


  // 2. 构造输入与输出，需要根据API的接口自定义构造

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, errorRet[0] = static_cast<float>(ret); return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, errorRet[0] = static_cast<float>(ret); return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, errorRet[0] = static_cast<float>(ret); return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, errorRet[0] = static_cast<float>(ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnAdd接口调用示例
  // 3. 调用CANN算子库API
  // 调用aclnnAdd第一段接口
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret);  errorRet[0] = static_cast<float>(ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);  errorRet[0] = static_cast<float>(ret); return ret);
  }
  // 调用aclnnAdd第二段接口
  ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); errorRet[0] = static_cast<float>(ret);  return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); errorRet[0] = static_cast<float>(ret);  return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); errorRet[0] = static_cast<float>(ret);  return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  std::copy(resultData.data(), resultData.data() + resultData.size(), dst);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

void aclnnAddTest(){
  std::vector<float> selfHostData(16384,1);
  std::vector<float> otherHostData(16384,1);
  float alphaValue = 1.2f;
  int len = 2;
  int width = 8192;
  std::vector<int64_t> selfShape = {len, width};
  std::vector<int64_t> otherShape = {len, width};
  std::vector<int64_t> outShape = {len, width};
  //int res = aclnnAddFunc(selfHostData, otherHostData, alphaValue, selfShape, otherShape, outShape);
}

