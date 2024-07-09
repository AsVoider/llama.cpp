#include <iostream>
#include <chrono>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnn-operator.h"
#include "aclnn-norm.h"
#include "aclnn-add.h"


int main() {

  int64_t ne1[4] = {3, 2, 2, 2};
  int64_t ne2[4] = {3, 2, 2, 1};
  int64_t ne[4] = {3, 3, 2, 2};
  float src1[24] = {
        0.0, 1.0, 2.0,      3.0, 4.0, 5.0,
        6.0, 7.0, 8.0,      9.0, 10.0, 11.0,

        12.0, 13.0, 14.0,   15.0, 16.0, 17.0,
        18.0, 19.0, 20.0,   21.0, 22.0, 23.0
    };
  float src2[12] = {
        0, 1, 1,  1, 1, 0,
        0, 1, 0,  1, 1, 1,
    };
  float dst[36];
  float pa[5] = {1.0, 8.0};
  auto start = std::chrono::high_resolution_clock::now();
  float scale = 1;

  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  aclnnGetRowsCompute(ne1, ne2, ne, src1, src2, dst, context, stream);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "running time: " << duration.count() << " ms" << std::endl;

  aclFloat16 A = 1.0;
  std::cout << "Elements of array b:" << std::endl;
  for (int i = 0; i < 36; ++i) {
    std::cout << dst[i] << " ";
  }
  std::cout << std::endl;


  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

