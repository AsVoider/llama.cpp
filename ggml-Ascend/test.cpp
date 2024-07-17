#include <iostream>
#include <vector>

#include "aclnn-compute.h"
#include "aclnn-math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnn-unary.h"
#include "common.h"
#include "aclnn-test.h"
#include "aclnn-comp.h"
#include "aclnn-mul.h"
#include "aclnn-operator.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0) 

/*

*/
#define aclnn_shape_t std::vector<int64_t>


void aclnnSoftMaxCompute(int64_t* ne, float* data, float* dst, float scale, aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> selfShape = {ne[0], ne[1], ne[2], ne[3]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  float dstTemp[ne[3]*ne[2]*ne[1]*ne[0]];
  int ret = aclnnMulsFunc(selfHostData, outHostData, scale, selfShape , outShape, dstTemp, context, stream);
  std::vector<int64_t> selfShape1 = {ne[0], ne[1], ne[2], ne[3]};
  std::vector<int64_t> outShape1 = selfShape1;
  std::vector<float> selfHostData1(dstTemp, dstTemp + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData1(selfHostData1.size(), 0);
  int ret1 = aclnnSoftMaxFunc(selfShape1, outShape1, selfHostData1, outHostData1, dst, context, stream);
}

int main() {
    int64_t ne1[4] = {4, 2, 2, 1};
    float data1[16];
    for (int i = 0; i<16 ;i++){
      data1[i]= i;
    } 
    float scale = 0.2;
    float dst[16];
      int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  aclrtContext context1 = nullptr;
  aclrtStream stream1 = nullptr;
  

    aclnnSoftMaxCompute(ne1, data1, dst , scale,context1, stream1 );

  float data2[32];
  for(int i = 0 ;i< 32; i++){
    data2[i] = i;
  }
  std::vector<int64_t> selfShape = {4, 5, 1, 1};
  std::vector<float> selfHostData(data2, data2+32);
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }




    return 0;
}

/*
  aclnn_pow_scalar_tensor_func
*/

// int main() {
//     // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {4, 2};
//     std::vector<int64_t> otherShape = {32, 16};
//     std::vector<int64_t> outShape = {4, 2};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData = {1, 1, 1, 2, 2, 2, 3, 3};
//     std::vector<float> otherHostData(512, 1);
//     std::vector<float> outHostData(8, 0);
// 
//     ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
// //     ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
// //     ctx->streams[deviceId][0] = stream;
// // 
// //     const int64_t ne[4] = {32, 16, 1, 1};
// // 
// //     ggml_tensor* src0 = new ggml_tensor();
// //     src0->ne[0] = ne[0];
// //     src0->ne[1] = ne[1];
// //     src0->ne[2] = ne[2];
// //     src0->ne[3] = ne[3];
// //     src0->type = GGML_TYPE_F32;
// //     src0->data = selfDeviceAddr;
// // 
// //     ggml_tensor* src1 = new ggml_tensor();
// //     src1->ne[0] = ne[1];
// //     src1->ne[1] = ne[0];
// //     src1->ne[2] = ne[2];
// //     src1->ne[3] = ne[3];
// //     src1->type = GGML_TYPE_F32;
// //     src1->data = otherDeviceAddr;
// // 
// //     ggml_tensor* dst = new ggml_tensor();
// //     dst->ne[0] = ne[1];
// //     dst->ne[1] = ne[1];
// //     dst->ne[2] = ne[2];
// //     dst->ne[3] = ne[3];
// //     dst->type = GGML_TYPE_F32;
// //     dst->data = outDeviceAddr;
// //     dst->src[0] = src0;
// //     dst->src[1] = src1;
// 
//     aclnn_pow_scalar_tensor_func((float) 1.2f, selfDeviceAddr, outDeviceAddr, selfShape, outShape, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, stream);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     auto size = GetShapeSize(outShape);
//     std::vector<float> resultData(size, 0);
//     ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//     for (int64_t i = 0; i < size; i++) {
//         LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
//     }
//     return 0;
// }

/*
  mul_mat
*/

// int main() {
//     // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {16, 32};
//     std::vector<int64_t> otherShape = {32, 16};
//     std::vector<int64_t> outShape = {16, 16};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData(512, 1);
//     std::vector<float> otherHostData(512, 1);
//     std::vector<float> outHostData(256, 0);
// 
//     ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
//     ctx->streams[deviceId][0] = stream;
// 
//     const int64_t ne[4] = {32, 16, 1, 1};
// 
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = ne[0];
//     src0->ne[1] = ne[1];
//     src0->ne[2] = ne[2];
//     src0->ne[3] = ne[3];
//     src0->type = GGML_TYPE_F32;
//     src0->data = selfDeviceAddr;
// 
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = ne[1];
//     src1->ne[1] = ne[0];
//     src1->ne[2] = ne[2];
//     src1->ne[3] = ne[3];
//     src1->type = GGML_TYPE_F32;
//     src1->data = otherDeviceAddr;
// 
//     ggml_tensor* dst = new ggml_tensor();
//     dst->ne[0] = ne[1];
//     dst->ne[1] = ne[1];
//     dst->ne[2] = ne[2];
//     dst->ne[3] = ne[3];
//     dst->type = GGML_TYPE_F32;
//     dst->data = outDeviceAddr;
//     dst->src[0] = src0;
//     dst->src[1] = src1;
// 
//     ggml_ascend_mul_mat(*ctx, src0, src1, dst);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     auto size = GetShapeSize(outShape);
//     std::vector<float> resultData(size, 0);
//     ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//     for (int64_t i = 0; i < size; i++) {
//         LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
//     }
//     return 0;
// }

/*
  RMS norm
*/

// int main() {
//     // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {2, 16};
//     std::vector<int64_t> otherShape = {4, 2};
//     std::vector<int64_t> outShape = {2, 16};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
//     std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
//     std::vector<float> outHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
// 
//     ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
//     ctx->streams[deviceId][0] = stream;
// 
//     const int64_t ne[4] = {16, 2, 1, 1};
// 
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = ne[0];
//     src0->ne[1] = ne[1];
//     src0->ne[2] = ne[2];
//     src0->ne[3] = ne[3];
//     src0->type = GGML_TYPE_F32;
//     src0->data = selfDeviceAddr;
// 
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = ne[0];
//     src1->ne[1] = ne[1];
//     src1->ne[2] = ne[2];
//     src1->ne[3] = ne[3];
//     src1->type = GGML_TYPE_F32;
//     src1->data = otherDeviceAddr;
// 
//     ggml_tensor* dst = new ggml_tensor();
//     dst->ne[0] = ne[0];
//     dst->ne[1] = ne[1];
//     dst->ne[2] = ne[2];
//     dst->ne[3] = ne[3];
//     dst->type = GGML_TYPE_F32;
//     dst->data = outDeviceAddr;
//     dst->src[0] = src0;
//     dst->src[1] = src1;
// 
//     ggml_ascend_rms_norm(*ctx, dst);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     auto size = GetShapeSize(outShape);
//     std::vector<float> resultData(size, 0);
//     ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//     for (int64_t i = 0; i < size; i++) {
//         LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
//     }
//     return 0;
// }

/*
  get_rows
*/

// int main() {
//     // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {2, 2, 2, 3};
//     std::vector<int64_t> otherShape = {1, 2, 2, 3};
//     std::vector<int64_t> outShape = {1, 1, 12, 3};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData = {
//         0.0, 1.0, 2.0,      3.0, 4.0, 5.0,
//         6.0, 7.0, 8.0,      9.0, 10.0, 11.0,
// 
//         12.0, 13.0, 14.0,   15.0, 16.0, 17.0,
//         18.0, 19.0, 20.0,   21.0, 22.0, 23.0
//     };
//     std::vector<int64_t> otherHostData = {
//         0, 1, 1,  1, 1, 0,
//         0, 1, 0,  1, 1, 1,
//     };
//     std::vector<float> outHostData(36, 0);
// 
//     ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
//     ctx->streams[deviceId][0] = stream;
// 
//     const int64_t ne[4] = {3, 2, 2, 2};
// 
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = ne[0];
//     src0->ne[1] = ne[1];
//     src0->ne[2] = ne[2];
//     src0->ne[3] = ne[3];
//     src0->type = GGML_TYPE_F32;
//     src0->data = selfDeviceAddr;
// 
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = 3;
//     src1->ne[1] = 2;
//     src1->ne[2] = 2;
//     src1->ne[3] = 1;
//     src1->type = GGML_TYPE_I64;
//     src1->data = otherDeviceAddr;
// 
//     ggml_tensor* dst = new ggml_tensor();
//     dst->ne[0] = ne[0];
//     dst->ne[1] = 3*2*2;
//     dst->ne[2] = 1;
//     dst->ne[3] = 1;
//     dst->type = GGML_TYPE_F32;
//     dst->data = outDeviceAddr;
//     dst->src[0] = src0;
//     dst->src[1] = src1;
// 
//     ggml_ascend_get_rows(*ctx, dst);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     auto size = GetShapeSize(outShape);
//     std::vector<float> resultData(size, 0);
//     ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//     for (int64_t i = 0; i < size; i++) {
//         LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
//     }
//     return 0;
// }

/*
  silu
*/

// int main() {
//     // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {4, 2};
//     std::vector<int64_t> otherShape = {4, 2};
//     std::vector<int64_t> outShape = {4, 2};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
//     std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
//     std::vector<float> outHostData(8, 0);
// 
//     ret = data_addr_malloc(selfShape, selfHostData, &selfDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(otherShape, otherHostData, &otherDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
//     ret = data_addr_malloc(outShape, outHostData, &outDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
//     ctx->streams[deviceId][0] = stream;
// 
//     const int64_t ne[4] = {2, 4, 1, 1};
// 
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = ne[0];
//     src0->ne[1] = ne[1];
//     src0->ne[2] = ne[2];
//     src0->ne[3] = ne[3];
//     src0->type = GGML_TYPE_F32;
//     src0->data = selfDeviceAddr;
// 
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = ne[0];
//     src1->ne[1] = ne[1];
//     src1->ne[2] = ne[2];
//     src1->ne[3] = ne[3];
//     src1->type = GGML_TYPE_F32;
//     src1->data = otherDeviceAddr;
// 
//     ggml_tensor* dst = new ggml_tensor();
//     dst->ne[0] = ne[0];
//     dst->ne[1] = ne[1];
//     dst->ne[2] = ne[2];
//     dst->ne[3] = ne[3];
//     dst->type = GGML_TYPE_F32;
//     dst->data = outDeviceAddr;
//     dst->src[0] = src0;
//     dst->src[1] = src1;
// 
//     ggml_ascend_silu(*ctx, dst);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     auto size = GetShapeSize(outShape);
//     std::vector<float> resultData(size, 0);
//     ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
//     for (int64_t i = 0; i < size; i++) {
//         LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
//     }
//     return 0;
// }