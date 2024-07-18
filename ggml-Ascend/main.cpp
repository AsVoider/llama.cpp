#include <iostream>
#include <chrono>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnn-operator.h"
#include "aclnn-norm.h"
#include "aclnn-add.h"
#include "aclnn-math.h"

// int main() {
// 
//   // ne = dst->src[0]->ne
//   // x = dst->src[0]->data
//   // dst = dst->data
//   // freq_scale = dst->op_params[6];
//   // freq_base = dst->op_params[5];
//   // n_dims = dst->op_params[1];
//   // pos = dst->src[1]->data
//   
//   auto start = std::chrono::high_resolution_clock::now();
// 
//   int32_t deviceId = 0;
//   aclrtContext context;
//   aclrtStream stream;
//   auto ret = Init(deviceId, &context, &stream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//   float dst[256];
//   int64_t ne[4]= {128,1,2,1};
//   int freq_scale = 2;
//   float freq_base = 1000;
//   int n_dims = 128;
//   int32_t pos[2] = {1, 2};
//   float x[256];
//   for (int i = 0; i<256; i++){
//     x[i]= 10000.0;
//   }
// 
//   aclnnRopeCompute(ne, freq_scale, freq_base, n_dims, pos, x, dst, context, stream);
// 
// 
// 
//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//   std::cout << "running time: " << duration.count() << " ms" << std::endl;
// 
//   std::cout << "Elements of array b:" << std::endl;
//   for (int i = 0; i < 36; ++i) {
//     std::cout << dst[i] << " ";
//   }
//   std::cout << std::endl;
// 
// 
//   aclrtDestroyStream(stream);
//   aclrtDestroyContext(context);
//   aclrtResetDevice(deviceId);
//   aclFinalize();
// 
//   return 0;
// }
// 
