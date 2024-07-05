#include <iostream>
#include <chrono>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnn-operator.h"
#include "aclnn-norm.h"
#include "aclnn-add.h"


int main() {

  int64_t ne1[4] = {1, 1, 1, 32};
  int64_t ne2[4] = {1, 1, 16, 1};
  float a[32] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float aa[16] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float b[32];
  float pa[5] = {1.0, 8.0};
  auto start = std::chrono::high_resolution_clock::now();
  float scale = 1;

  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  aclnnAddCompute(ne1, ne2, a, aa, b, context, stream);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "running time: " << duration.count() << " ms" << std::endl;

  aclFloat16 A = 1.0;
  std::cout << "Elements of array b:" << std::endl;
  for (int i = 0; i < 32; ++i) {
    std::cout << b[i] << " ";
  }
  std::cout << std::endl;


  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

