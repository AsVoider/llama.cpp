#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnn-comp.h"
#include "aclnnop/aclnn_softmax.h"
#include "aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h"

int aclnn_soft_max_func(void* selfDataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType outDataType,
  aclrtStream &stream) {
  
  auto ret = 0;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  int64_t dim = 0;

  ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmax failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(self);
  aclDestroyTensor(out);

  return 0;
}

int aclnn_soft_max_func(void * dataAddr, void * maskAddr, float scale, void * outAddr,
    aclnn_shape_t & dataShape, aclnn_shape_t & maskShape, aclnn_shape_t & outShape,
    aclDataType dataType, aclDataType maskType, aclDataType outType, aclrtStream & stream) {
  auto ret(0);

  aclTensor * dataTensor(nullptr);
  ret = create_acl_tensor(dataShape, dataType, &dataAddr, &dataTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor * maskTensor(nullptr);
  ret = create_acl_tensor(maskShape, maskType, &maskAddr, &maskTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor * outTensor(nullptr);
  ret = create_acl_tensor(outShape, outType, &outAddr, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  aclTensor * biasTensor(nullptr);
  void * biasAddr(nullptr);
  int64_t sz(dataShape[0] * dataShape[1] * dataShape[2] * dataShape[3]);
  ret = aclrtMalloc(&biasAddr, sz * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemset(biasAddr, sz * sizeof(float), 0, sz * sizeof(float));
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // std::vector<float> bias(sz, 0);
  aclnn_shape_t biasShape{dataShape[0], dataShape[1], dataShape[2], dataShape[3]};
  ret = create_acl_tensor(biasShape, outType, &biasAddr, &biasTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // {
  //   printf("datatype is %d, masktype is %d, outtype is %d\n", dataType, maskType, outType);
  //   float d[32], m[32], o[32];
  //   aclrtMemcpy((void *)d, 32 * 4, dataAddr, 32 * 4, ACL_MEMCPY_DEVICE_TO_HOST);
  //   aclrtMemcpy((void *)m, 32 * 4, maskAddr, 32 * 4, ACL_MEMCPY_DEVICE_TO_HOST);
  //   for (auto dt : d) {
  //     printf("%f ", dt);
  //   }
  //   printf("\n");
  //   for (auto mt : m) {
  //     printf("%f ", mt);
  //   }
  //   printf("\n");
  //   // aclrtMemcpy((void *)d, 32 * 4, dataAddr, 32 * 4, ACL_MEMCPY_DEVICE_TO_HOST);
  //   for (auto &sp : dataShape) {
  //       printf("%ld ", sp);
  //   }
  //   printf("\n");
  //   for (auto &sp : maskShape) {
  //       printf("%ld ", sp);
  //   }
  //   printf("\n");
  //   for (auto &sp : biasShape) {
  //       printf("%ld ", sp);
  //   }
  //   printf("\n");
  //   for (auto &sp : outShape) {
  //       printf("%ld ", sp);
  //   }
  //   printf("\n");
  //   printf("sacle is %f\n", scale);
  // }

  uint64_t workSpaceSize(0);
  aclOpExecutor * exe;

  ret = aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(dataTensor, maskTensor, biasTensor, scale, 0, outTensor, &workSpaceSize, &exe); // for test
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed at get sm workspace: %d %s\n", ret, aclGetRecentErrMsg());return ret);

  void* workspaceAddr = nullptr;
  if (workSpaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMaskedSoftmaxWithRelPosBias(workspaceAddr, workSpaceSize, exe, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(dataTensor);
  aclDestroyTensor(maskTensor);
  aclDestroyTensor(biasTensor);
  aclDestroyTensor(outTensor);
  return ret;
}

int acl_soft_max_func(void* selfDataAddr, void* outDataAddr, aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType otherDataType, aclDataType outDataType,
  aclrtStream &stream){

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  auto ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  int64_t dim = 0;

  ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmax failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(self);
  aclDestroyTensor(out);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}


int aclnnSoftMaxFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream ){
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  // 创建self aclTensor
  int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  int64_t dim = 0;
  // 调用aclnnSoftmax第一段接口
  ret = aclnnSoftmaxGetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSoftmax第二段接口
  ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmax failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

void aclnnSoftMaxTest(){
  std::vector<int64_t> selfShape = {4, 1 ,2, 1};
  std::vector<int64_t> outShape = {4, 1 ,1, 2};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float a[200];
  //int ret = aclnnSoftMaxFunc(selfShape, outShape, selfHostData, outHostData, a);
}