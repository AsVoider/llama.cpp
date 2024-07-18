#include <iostream>
#include <chrono>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnn-operator.h"
#include "aclnn-norm.h"
#include "aclnn-add.h"
#include "aclnn-math.h"
#include "aclnn-compute.h"

int main() {

  // ne = dst->src[0]->ne
  // x = dst->src[0]->data
  // dst = dst->data
  // freq_scale = dst->op_params[6];
  // freq_base = dst->op_params[5];
  // n_dims = dst->op_params[1];
  // pos = dst->src[1]->data


  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);


  ret = aclrtCreateStream(&stream);

  // 1. 申请内存
uint64_t size = 1 * 1024 * 1024;
void* hostAddr = NULL;
void* devAddr = NULL;
aclrtMallocHost(&hostAddr, size + 64);
aclrtMalloc(&devAddr, size, ACL_MEM_MALLOC_NORMAL_ONLY);

// 2. 异步内存复制
// aclrtStream stream = NULL;
// 申请内存后，可向内存中读入数据，该自定义函数ReadFile由用户实现
ret = aclrtMemcpy(devAddr, size, hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
if (ret != ACL_SUCCESS) {
  std::cout<< aclGetRecentErrMsg()<< std::endl;
  exit(0);
}
  
ret = aclrtSynchronizeStream(stream);
if (ret != ACL_SUCCESS) {
  std::cout<< "fuck\n" << aclGetRecentErrMsg()<< std::endl;
}
  
// 3. 释放资源
aclrtDestroyStream(stream);
aclrtFreeHost(hostAddr);
aclrtFree(devAddr);

  return 0;
}

