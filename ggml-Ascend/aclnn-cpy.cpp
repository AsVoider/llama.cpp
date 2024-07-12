#include "common.h"
#include "aclnn-cpy.h"
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_index_select.h"

int aclnn_cpy_func(void* selfRefDataAddr, void* srcDataAddr,
    aclnn_shape_t& selfRefShape, aclnn_shape_t& srcShape,
    aclDataType selfRefDataType, aclDataType srcDataType,
    aclrtStream &stream) {
    
    auto ret = 0;

    aclTensor* selfRef = nullptr;
    aclTensor* src = nullptr;

    ret = create_acl_tensor(selfRefShape, selfRefDataType, &selfRefDataAddr, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = create_acl_tensor(srcShape, srcDataType, &srcDataAddr, &src);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnInplaceCopyGetWorkspaceSize(selfRef, src, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(selfRef);
    aclDestroyTensor(src);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

int aclnnCpyFunc(std::vector<int64_t>& selfRefShape, std::vector<int64_t>& srcShape, std::vector<float>& selfRefHostData, std::vector<float>& srcHostData,  float* dst, aclrtContext &context, aclrtStream &stream){
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* selfRefDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* src = nullptr;
  // 创建selfRef aclTensor
  auto ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceCopy第一段接口
  ret = aclnnInplaceCopyGetWorkspaceSize(selfRef, src, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceCopy第二段接口
  ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  std::copy(resultData.data(), resultData.data() + resultData.size(), dst);
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(src);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(srcDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int aclnn_get_rows_func(void* selfDataAddr, void* indexDataAddr, void* outDataAddr,
	aclnn_shape_t& selfShape, aclnn_shape_t& indexShape, aclnn_shape_t& outShape,
	aclDataType selfDataType, aclDataType indexDataType, aclDataType outDataType,
	aclrtStream &stream) {
	
	auto ret = 0;

	aclTensor* self = nullptr;
	aclTensor* index = nullptr;
	aclTensor* out = nullptr;

	int64_t dim = 2;

	ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	ret = create_acl_tensor(indexShape, indexDataType, &indexDataAddr, &index);
	CHECK_RET(ret == ACL_SUCCESS, return ret);
	ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
	CHECK_RET(ret == ACL_SUCCESS, return ret);

	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;

	ret = aclnnIndexSelectGetWorkspaceSize(self, dim, index, out, &workspaceSize, &executor);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelectGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
	void* workspaceAddr = nullptr;
	if (workspaceSize > 0) {
		ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
		CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
	}
	ret = aclnnIndexSelect(workspaceAddr, workspaceSize, executor, stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelect failed. ERROR: %d\n", ret); return ret);

	ret = aclrtSynchronizeStream(stream);
	CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

	aclDestroyTensor(self);
	aclDestroyTensor(index);
	aclDestroyTensor(out);

	if (workspaceSize > 0) {
		aclrtFree(workspaceAddr);
	}

	return 0;
}

int aclnnGetRowsFunc(  std::vector<int64_t> &selfShape,
  std::vector<int64_t> &indexShape,
  std::vector<int64_t> &outShape,
  int64_t dim,
  std::vector<float> &selfHostData,
  std::vector<int> &indexHostData,
  std::vector<float> &outHostData,float* dst, aclrtContext &context, aclrtStream &stream){
  
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* out = nullptr;

  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnIndexSelect第一段接口
  ret = aclnnIndexSelectGetWorkspaceSize(self, dim, index, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelectGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnIndexSelect第二段接口
  ret = aclnnIndexSelect(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexSelect failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(index);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}