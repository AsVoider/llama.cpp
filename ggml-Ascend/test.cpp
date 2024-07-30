#include <iostream>
#include <vector>
#include <cstring>

#include "aclnn-compute.h"
#include "aclnn-math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

// int main() {
//     // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
//     // 根据自己的实际device填写deviceId
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     auto ret = Init(deviceId, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {2, 2, 2, 1};
//     std::vector<int64_t> otherShape = {2, 3, 1, 1};
//     std::vector<int64_t> outShape = {2, 2, 2, 1};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
//     std::vector<float> otherHostData = {1, 1, 1, 1, 1, 1};
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
//     const int64_t ne[4] = {2, 2, 2, 1};
// 
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = ne[0];
//     src0->ne[1] = ne[1];
//     src0->ne[2] = ne[2];
//     src0->ne[3] = ne[3];
//     src0->nb[0] = sizeof(float);
//     src0->nb[1] = src0->nb[0] * src0->ne[0];
//     src0->nb[2] = src0->nb[1] * src0->ne[1];
//     src0->nb[3] = src0->nb[2] * src0->ne[2];
//     src0->type = GGML_TYPE_F32;
//     src0->data = selfDeviceAddr;
// 
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = 2;
//     src1->ne[1] = 3;
//     src1->ne[2] = 1;
//     src1->ne[3] = 1;
//     src1->type = GGML_TYPE_F32;
//     src1->data = otherDeviceAddr;
// 
//     ggml_tensor* dst = new ggml_tensor();
//     dst->ne[0] = ne[0];
//     dst->ne[1] = ne[1];
//     dst->ne[2] = ne[2];
//     dst->ne[3] = ne[3];
//     dst->nb[0] = sizeof(float);
//     dst->nb[1] = dst->nb[0] * dst->ne[0];
//     dst->nb[2] = dst->nb[1] * dst->ne[1];
//     dst->nb[3] = dst->nb[2] * dst->ne[2];
//     dst->type = GGML_TYPE_F32;
//     dst->data = outDeviceAddr;
//     dst->src[0] = src0;
//     dst->src[1] = src1;
//     dst->op_params[0] = 1.0f;
// 
//     ggml_ascend_cpy(*ctx, src0, dst);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     LOG_PRINT("\nresult: \n");
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
  soft max
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
//     std::vector<int64_t> selfShape = {2, 2, 2, 1};
//     std::vector<int64_t> otherShape = {2, 3, 1, 1};
//     std::vector<int64_t> outShape = {2, 2, 2, 1};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
//     std::vector<float> otherHostData = {1, 1, 1, 1, 1, 1};
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
//     const int64_t ne[4] = {2, 2, 2, 1};
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
//     src1->ne[0] = 2;
//     src1->ne[1] = 3;
//     src1->ne[2] = 1;
//     src1->ne[3] = 1;
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
//     dst->op_params[0] = 1.0f;
// 
//     ggml_ascend_soft_max(*ctx, dst);
//     CHECK_RET(ret == ACL_SUCCESS, return ret);
// 
//     LOG_PRINT("\nresult: \n");
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
  rope
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
//     std::vector<int64_t> selfShape = {1, 2, 1, 128};
//     std::vector<int64_t> otherShape = {2};
//     std::vector<int64_t> outShape = {16, 16};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<float> selfHostData(256, 10000.0);
//     std::vector<int32_t> otherHostData = {1, 2};
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
//     const int64_t ne[4] = {128, 1, 2, 1};
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
//     src1->type = GGML_TYPE_I32;
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
//     dst->op_params[5] = 1000;
//     dst->op_params[6] = 2;
//     dst->op_params[1] = 128;
// 
//     ggml_ascend_rope(*ctx, dst);
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
//     const int64_t ne0[4] = {4, 8, 2, 1};
//     const int64_t ne1[4] = {4, 8, 8, 1};
//     const int64_t ne[4] = {8, 8, 8, 1};
// 
//     // 2. 构造输入与输出，需要根据API的接口自定义构造
//     std::vector<int64_t> selfShape = {ne0[3], ne0[2], ne0[1], ne0[0]};
//     std::vector<int64_t> otherShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
//     std::vector<int64_t> outShape = {ne[3], ne[2], ne[1], ne[0]};
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     aclTensor* self = nullptr;
//     aclTensor* other = nullptr;
//     aclScalar* alpha = nullptr;
//     aclTensor* out = nullptr;
//     std::vector<aclFloat16> selfHostData(ne0[0] * ne0[1] * ne0[2] * ne0[3], aclFloatToFloat16(1.0));
//     std::vector<float> otherHostData(ne1[0] * ne1[1] * ne1[2] * ne1[3], 1.0);
//     std::vector<float> outHostData(ne[0] * ne[1] * ne[2] * ne[3], 0.0);
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
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = ne0[0];
//     src0->ne[1] = ne0[1];
//     src0->ne[2] = ne0[2];
//     src0->ne[3] = ne0[3];
//     src0->type = GGML_TYPE_F16;
//     src0->data = selfDeviceAddr;
// 
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = ne1[0];
//     src1->ne[1] = ne1[1];
//     src1->ne[2] = ne1[2];
//     src1->ne[3] = ne1[3];
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

// int main() {
//     int32_t deviceId = 0;
//     aclrtStream stream;
//     aclrtContext context = nullptr;
//     auto ret = Init(deviceId, &context, &stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));
//     ggml_backend_ascend_context* ctx = new ggml_backend_ascend_context(deviceId);
//     ctx->streams[deviceId][0] = stream;
// 
//     ggml_tensor* dst = new ggml_tensor();
//     dst->ne[0] = 4096;
//     dst->ne[1] = 1;
//     dst->ne[2] = 1;
//     dst->ne[3] = 1;
//     ggml_tensor* src0 = new ggml_tensor();
//     src0->ne[0] = 4096;
//     src0->ne[1] = 2;
//     src0->ne[2] = 1;
//     src0->ne[3] = 1;
//     ggml_tensor* src1 = new ggml_tensor();
//     src1->ne[0] = 1;
//     src1->ne[1] = 1;
//     src1->ne[2] = 1;
//     src1->ne[3] = 1;
//     dst->src[0] = src0;
//     dst->src[1] = src1;
//     // float freq_base = 10000.0;
//     // float freq_scale = 1.0;
//     // dst->op_params[1] = 128;
//     // memcpy((int32_t *)dst->op_params + 5, &freq_base, sizeof(float));
//     // memcpy((int32_t *)dst->op_params + 6, &freq_scale, sizeof(float));
// 
//     std::vector<float> src0data = {0.034739, 0.147054, -0.082419, 0.011228, -0.015383, -0.078130, -0.378723, -0.055443, -0.059841, -0.056598, -0.199576, -0.046264, 0.180794, 0.102280, -0.140968, -0.013434, -0.142752, 0.069908, 0.124636, 0.065206, 0.362012, 0.098319, -0.147525, 0.212681, 0.052879, 0.074995, -0.003999, -0.181311, 0.161409, 0.410361, -0.148599, -0.191808, -0.115645, -0.000080, 0.105427, 0.382512, 0.126079, -0.226366, 0.175497, -0.005549, 0.173348, 0.315642, 0.259557, 0.111927, -0.317313, -0.005636, -0.574168, -0.171177, -0.397105, 0.139389, -0.112327, 0.519882, -0.170515, -0.244640, -0.273431, 0.150586, 0.037725, -0.279003, 0.160863, -0.313984, -0.044864, -0.196678, 0.043581, 0.017215, 0.110621, -0.077227, -0.243094, 0.357249, 0.031164, 0.087661, -0.358108, -0.146817, 0.088849, 0.178415, -0.155272, 0.031867, -0.166640, -0.157140, -0.119449, -0.089515, -0.155147, 0.411057, -0.315951, -0.141397, -0.013909, 0.328771, -0.103104, -0.191605, 0.278597, -0.209802, 0.065479, -0.028876, 0.432954, 0.243501, 0.657622, -0.135483, -0.202176, 0.000090, -0.066743, -0.038875, 0.299605, 0.580528, -0.022468, 0.115778, 0.088607, -0.027510, -0.089402, 0.265931, 0.041976, -0.172565, -0.210584, 0.568348, -0.014722, -0.022051, 0.296592, -0.194104, -0.035237, 0.113478, 0.192730, 0.095969, 0.068482, -0.411696, -0.211155, -0.017572, -0.326554, 0.015024, 0.246404, -0.137512, 0.029340, -0.368214, -0.229044, -0.075909, -0.305440, -0.069287, -0.132193, -0.038594, -0.267999, -0.004354, -0.514653, 0.220183, -0.437222, 0.166319, 0.063213, -0.118082, 0.092751, -0.277428, 0.231172, -0.012276, -0.069730, -1.118826, 0.171251, 0.296167, -0.035069, -0.106553, -0.229955, 0.499489, 0.282908, -0.143722, -0.122650, -0.024268, -0.056459, -0.009790, -0.040710, 0.092550, 0.137879, -0.212828, 0.382874, 0.204012, 0.212198, 0.032986, -0.090788, 0.115304, 0.541186, -0.122621, -0.208514, 0.234676, -0.210997, -0.079895, -0.100833, 0.098987, 0.473402, 0.033026, -0.173302, -0.303470, -0.273292, 0.105827, 0.019240, -0.404203, 0.089140, 0.440216, 0.231690, 0.152083, 0.160171, 0.192648, 0.165381, -0.184427, -0.446022, 0.213179, -0.081188, 0.217865, -0.290683, -0.261126, 0.120336, 0.040275, 0.158938, 0.176430, -0.011388, -0.216045, 0.175942, 0.227663, 0.019233, 0.280092, 0.152573, 0.155663, 0.046448, -0.041987, 0.157639, 0.039512, 0.206021, -0.169790, -0.061047, -0.202348, 0.265485, -0.031616, 0.197199, -0.143519, 0.130985, -0.050378, -0.281923, 0.090536, 0.250100, 0.168531, -0.288150, 0.334228, 0.205976, -0.226475, -0.297202, 0.194683, 0.063446, -0.321010, -0.020351, 0.081132, 0.048455, -0.067786, 0.283482, -0.042100, -0.335245, 0.038412, -0.288651, 0.140930, 0.149108, 0.194883, 0.321428, -0.168348, 0.100112, -0.117906, -0.010011, -0.682903, -0.076815, 0.463831, -0.131928, 0.167414, -0.171953, -0.460725, 0.026846, 0.006684, 0.118026, -0.015334, 0.277564, -0.411332, 0.128679, -0.403444, 0.149967, 0.223266, -0.303678, -0.006968, -0.214844, 0.002572, 0.044901, -0.016977, 0.393908, 0.190995, 0.202054, 0.097353, -0.327485, -0.201753, -0.037329, -0.273365, -0.186316, 0.071924, 0.334652, -0.050978, -0.020696, -0.100858, 0.127633, 0.287501, -0.182810, 0.154396, -0.206823, 0.249646, -0.082631, -0.207490, 0.239357, 0.097713, 0.017332, -0.000418, -0.048251, 0.327647, 0.282937, 0.374449, -0.600304, 0.307485, -0.426558, -0.026864, 0.247150, -0.229972, 0.019194, 0.133859, 0.231860, -0.089240, 0.064662, 0.243742, -0.068991, 0.299906, 0.163395, 0.002712, -0.138154, 0.132886, -0.219913, -0.012319, -0.127290, 0.171162, -0.092079, -0.141956, 0.052590, -0.065020, 0.024167, -0.311664, 0.138411, -2.709367, -0.115516, 0.158059, -0.083855, 0.410526, 0.058950, -0.023074, 0.299442, -0.219639, 0.025806, -0.228876, -0.183229, -0.317094, -0.545534, 0.093345, -0.387790, -0.006804, -0.274341, -0.341168, -0.083632, -0.077075, -0.327006, 0.092271, -0.154387, -4.690395, 0.011700, -0.060964, -0.069358, -0.244574, -0.124052, -0.265086, -0.195603, -0.097297, 0.084795, -0.147511, -0.222899, 0.162092, 0.094606, -0.230729, 0.172643, -0.150422, -0.190172, 0.061213, 0.100283, 0.216296, 0.279293, 0.044750, 0.342544, 0.059411, -0.287774, 0.091408, 0.151986, 0.081550, 0.428838, -0.167801, -0.055635, -0.135711, 0.205098, 0.059540, -0.288254, -0.242965, 0.097126, 0.267838, -0.066279, -0.205817, -0.459267, -0.022740, 0.049728, 0.095725, 0.210846, -0.519307, 0.212310, -0.391347, -0.160900, 0.072044, -0.037480, -0.030729, 0.295801, 0.176222, -0.210070, -0.003678, -0.242986, 0.113814, -0.385129, -0.348500, 0.032723, 0.409280, -0.154869, -0.169100, 0.545317, -0.175855, -0.064652, -0.055591, -0.097611, 0.115774, 0.128695, 0.160132, -0.068493, 0.346441, -0.074979, 0.269519, -0.412793, 0.213414, -0.331867, -0.182466, -0.081598, 0.139770, -0.086419, 0.161506, 0.451153, -0.275372, 0.313056, 0.002777, 0.351072, -0.212279, 0.035056, -0.022684, -0.147530, 0.005010, 0.199532, 0.207610, -0.429153, -0.205635, 0.431581, -0.417790, 0.232122, -0.055064, -0.292535, -0.198574, 0.246411, 0.141015, -0.749484, 0.077992, -0.169253, -0.281335, -0.080153, 0.521327, -0.048307, -0.068122, 0.073250, -0.096746, 0.110064, -0.175960, 0.227207, 0.156167, 0.020087, -0.374797, 0.175857, -0.212817, 0.113664, -0.111318, -0.249355, -0.086341, -0.009884, 0.040205, -0.271140, -0.162672, 0.221104, -0.083640, -0.005092, 0.182303, -0.106867, 0.439903, 0.119376, 0.010087, -0.125609, 0.017507, 0.086714, 0.173541, 0.395370, 0.196424, -0.071768, 0.240140, 0.270737, 0.204910, -0.034699, 0.126404, 0.170566, -0.267178, 0.244291, -0.150556, 0.174465, -0.044934, -0.236289, -0.364022, 0.002693, -0.521671, -0.146650, -0.132931, 0.212912, 0.077528, 0.004135, -0.080428, -0.225663, -0.173864, -0.179997, 0.083741, 0.012553, -0.239107, -0.063342, -0.381712, 0.067973, -0.042569, -0.051023, 0.054272, -0.103258, 0.032879, 0.053066, -0.111126, -0.137830, 0.044814, -0.178044, 0.009395, -0.074107, 0.137550, 0.299134, 0.114164, 0.017964, 0.012354, -0.010717, 0.194762, -0.208757, 0.138536, 0.057631, 0.517905, -0.232968, 0.317784, -0.045333, -0.055482, -0.061115, 0.165026, -0.032016, 0.065376, 0.329684, -0.101565, 0.050220, 1.052310, -0.093452, 0.076090, -0.228670, -0.026164, 0.734430, -0.293868, -0.111881, -0.410658, -0.239001, 0.165276, -0.164192, 0.154541, 0.108861, 0.725170, 0.120251, -0.143344, 0.025223, 0.140371, -0.334887, 0.269681, -0.187700, 0.232172, 0.064140, -0.149756, -0.148315, 0.254425, -0.081046, -0.021029, -0.432093, 0.014382, 0.005159, -0.381092, -0.233551, -0.097668, 0.113308, 0.143564, -0.088765, 0.172120, 0.223122, -0.175462, 0.325728, 0.016280, 0.045174, -0.035029, -0.074545, 0.294218, 0.329079, 0.017529, 0.158083, 0.012038, 0.019914, 0.088863, -0.181363, -0.317748, -0.100366, 0.243206, -0.021411, 0.004314, -0.503220, -0.243928, -0.011712, -0.161674, -0.080567, 0.235169, -0.234753, 0.130056, -0.095914, 0.048622, 0.121430, 0.095835, -0.465570, -0.375675, 0.125060, -0.175570, 0.136043, -0.067120, 0.255253, 0.401066, 0.304688, 0.236636, -0.190327, -0.388932, 0.063165, 0.391291, 0.069176, -0.141437, -0.236398, -0.271327, 0.127421, 0.101836, 0.081815, -0.208991, -0.154148, 0.358901, -0.485662, -0.067637, -0.172556, -0.226096, 0.036498, 0.310673, -0.109069, 0.151214, 0.132796, -0.326450, -0.264099, -0.076373, 0.262908, 0.127677, 0.124780, -0.105347, -0.021151, -0.077353, 0.267011, 0.207759, 0.070948, -0.068634, -0.119484, 0.287602, -0.202578, -0.274728, 0.256984, -0.124350, -0.409791, -0.089536, 0.311397, 0.084458, 0.095087, 0.008410, 0.159674, 0.008352, -0.118152, 0.250844, -0.316234, -0.223501, 0.180965, 0.047952, 0.009672, -0.208610, -0.055305, -0.073090, -0.169984, 0.194815, 0.266289, -0.130619, 0.126372, 0.095787, -0.238265, -0.105074, -0.293805, 0.014590, -0.084911, 0.606066, -0.100421, 0.130767, 0.075911, 0.111268, 0.266035, -0.286911, 0.294457, 0.047351, 0.159590, -0.176327, -0.163036, 0.378033, 0.451396, 0.025010, 0.411066, -0.527410, 0.208401, 0.247025, -0.033245, -0.204583, -0.141464, -0.072414, -0.240975, -0.265123, 0.456201, -0.009050, 0.211890, 0.119972, 0.192359, 0.012523, -0.102318, -0.283177, -0.064632, 0.046949, 0.045833, 0.001507, -0.396025, 0.126001, -0.202520, -0.238200, 0.365791, -0.336802, 0.175474, 0.190387, -0.047578, -0.144892, -0.079547, -0.064927, -0.140325, 0.017589, 0.146349, -0.269913, -0.013659, -0.252820, -0.033482, 0.045839, 0.210123, 0.164476, 0.103349, -0.157470, -3.928395, 0.049556, 0.051157, -0.076161, -0.153714, 0.342558, 0.035315, -0.173146, -0.107874, -0.289053, -0.422082, 0.005065, 0.087103, -0.027430, -0.305831, -0.046754, -0.220575, -0.075693, -0.133282, -0.189152, -0.119736, -0.560760, -0.281359, -0.162736, -0.108119, 0.302449, -0.234032, -0.084981, 0.009093, 0.141931, 0.054809, 0.065009, -0.090365, -0.065891, 0.380499, 0.176504, -0.362192, -0.199244, 0.048586, -0.134074, -0.019720, -0.125119, 0.103863, -0.192932, -0.202767, 0.041233, 0.041685, 0.000708, -0.152788, -0.154694, 0.118554, 0.114857, 0.034730, 0.106878, 0.209857, -0.306223, -0.042450, 0.243489, 0.038933, 0.059054, 0.265883, 0.025619, 0.086444, 0.163985, -0.505833, -0.058469, 0.308504, -0.212087, 0.229047, 0.123431, 0.455860, 0.132026, 0.281572, -0.077174, 0.357248, 0.150100, 0.150837, 0.242590, 0.014367, 0.044119, -0.001074, -0.131837, -0.144705, -0.207571, 0.123544, 0.088904, 0.144702, 0.127541, 0.038358, 0.294218, 0.393512, -0.008953, 0.092855, 0.107910, 0.157726, -0.161359, 0.185141, -0.206368, 0.050208, -0.000120, -0.178317, 0.072704, 0.130612, -0.025483, -0.277605, -0.239696, -0.002480, -0.431593, -0.368146, 0.133848, 0.010432, 0.337014, -0.058737, -0.375947, 0.084092, -0.182888, -0.203264, -0.110575, 0.036279, -0.368752, 0.337900, -0.020584, -0.009611, -0.068938, -0.030336, -0.459904, -0.165515, 0.004111, 0.372891, 0.287551, 0.070984, 0.094500, -0.022945, 0.019036, -0.225854, 0.347151, -0.106370, 0.130824, -0.378250, 0.053342, 0.081912, 0.130940, -0.168690, 0.232440, 0.134731, -0.106594, -0.151832, -0.168529, 0.028148, 0.218289, 0.120038, 0.100965, -0.143209, -0.320700, -0.153985, 0.058591, 0.162447, -0.318080, -0.183066, 0.345189, 0.328525, -0.046204, 0.139827, -0.104075, -0.350510, 0.072639, -0.005154, 0.117596, 0.087359, -0.092254, 0.232115, -0.041776, 0.208919, -0.178784, 0.040997, 0.049447, 0.336325, 0.139624, 0.088888, -0.045969, -0.442140, 0.125613, -0.304115, -0.123363, -1.287873, 0.132547, 0.113948, 0.058852, 0.103720, 0.057739, 0.071578, 0.110356, -0.149405, 0.270006, 0.503441, 0.058378, -0.105486, 0.099244, 0.192790, -0.473245, -0.159269, -0.562536, 0.333486, -0.383086, 0.095237, 0.048344, -0.415301, 0.198240, -0.257927, 0.110142, 0.304047, -0.270725, -0.050761, 0.024315, -0.083589, -0.333560, 0.027873, 0.313629, -0.266251, -0.262524, 0.112304, 0.066363, -0.151897, -0.187903, -0.182553, 0.206915, 0.153070, -0.241537, -0.406492, 0.089053, -0.159859, -0.252732, -0.015935, -0.011696, 0.269137, -0.377608, -0.381571, 0.100829, 0.276266, 0.000146, 0.255833, -0.159419, 0.067980, 0.156262, 0.275155, 0.141051, 0.462160, 0.184755, 0.050219, 0.077573, -0.061278, -0.210477, 0.445022, 0.284457, 0.100658, -0.170611, -0.104690, -0.102377, -0.142149, 0.117870, -0.082378, -0.299336, 0.017991, 0.190760, -0.494981, -0.226074, -0.059314, 0.123437, 0.321820, 0.386834, 0.186884, 0.043763, -0.475047, 0.228042, -0.213867, -0.094696, -0.291995, 0.187610, 0.119034, -0.155390, -0.051862, -0.063873, 0.213712, 0.080719, 0.019040, 0.020606, -0.168523, 0.206653, 8.170330, -0.190862, 0.014350, 0.027357, -0.065716, 0.051146, -0.011765, 0.214958, -0.206184, -0.007959, 0.031964, 0.003138, 0.240017, -0.171486, -0.003216, -0.122397, 0.029250, -0.102432, -0.398519, -0.130438, 0.000740, -0.138126, -0.131905, -0.135074, -0.286922, -0.060559, -0.017670, -0.207687, -0.018401, -0.063516, 0.183546, -0.193253, 0.006446, -0.006792, -0.170223, -0.030363, 0.155102, 0.349276, -0.041016, -0.109138, 0.166073, 0.089460, 0.243911, 0.253298, -0.140366, -0.118813, 0.050544, -0.058683, -0.131330, -0.004431, -0.007045, -0.129216, 0.029276, 0.301354, 0.191122, 0.472362, -0.401065, 0.200203, 0.334611, 0.186905, -0.049818, -0.338806, 0.209490, -0.460105, 0.107425, 0.090164, 0.066156, -0.047555, -0.079601, -0.218673, -0.071366, 0.032928, 0.005940, -0.227324, 0.142054, 0.236613, 0.087574, -0.044356, 0.045784, 0.026395, -0.137234, 0.132712, 0.078597, 0.278580, -0.246688, -0.243624, -0.153059, 0.010236, -0.219449, -0.264090, -0.258024, 0.410952, 0.119434, 0.240904, 0.181252, 0.068467, 0.021258, -0.204323, 0.251762, -0.039519, -0.620016, -0.188084, 0.100205, 0.328695, -0.067050, 0.065681, -0.279265, 0.047715, -0.182035, 0.015696, 0.179247, 0.418379, 0.338081, 0.533580, 0.286481, -0.443341, 0.060699, 0.045450, -0.086932, -0.037020, 0.157577, 0.023656, -0.097264, 0.101469, -0.329366, -0.085430, -0.068128, -0.057109, -0.034191, 0.008105, 0.224700, -0.210472, -0.111570, 0.219028, -0.082220, 0.102978, 0.161371, 0.409997, 0.094485, 0.212825, -0.003942, 0.301771, 0.574425, 0.084680, -0.296989, -0.009510, -0.240003, -0.121150, -0.305246, -0.232538, -0.088973, 0.210916, -0.059335, -0.249472, -0.118923, -0.096416, 0.069355, 0.368272, -0.046450, -0.074801, -0.130568, -0.062984, 0.118581, 0.198835, -0.044209, 0.378823, -0.156550, 0.144820, -0.275972, -0.252151, -0.292486, -0.327898, 0.136953, 0.148258, 0.085189, 0.265386, -0.008169, 0.173049, 0.567923, 0.110858, 0.040368, 0.018324, -0.209357, -0.410602, 0.025011, -0.011769, -0.201580, -0.089199, -0.111341, -0.235116, -0.321254, 0.062037, 0.195370, -0.322261, -0.174689, 0.190846, -0.388379, -0.043615, -0.097401, 0.060351, 0.051849, -0.172534, 0.029780, -0.010820, 0.020008, -0.198275, 0.043929, 0.200785, 0.172604, 0.269945, 0.105885, -0.106413, 0.025159, 0.221841, -0.049681, -0.000925, -0.045690, -0.090897, 0.053544, 0.124369, -0.096088, -0.265087, -0.048779, -0.255643, 0.184760, 0.586788, 0.081949, 0.219830, -0.077057, 0.192512, 0.260177, 0.274981, -0.080219, 0.026956, -0.061562, 0.122835, 0.182197, -0.024402, -0.036334, 0.099140, 0.107783, 0.038798, -0.170515, -0.505156, -0.175998, 0.025282, -0.369270, 0.142937, -0.219616, 0.202360, 0.107114, 0.153912, 0.156011, -0.379004, -0.225620, 0.079456, -0.157785, 0.157906, 0.052077, 0.381034, -0.114714, -0.173286, -0.749749, -0.345531, -0.103142, 0.159966, -0.043536, 0.204427, 0.101691, 0.064427, 0.500658, 0.022466, -0.039095, 0.016137, -0.411802, -0.230247, -0.183436, 0.493457, 0.144697, -0.056397, 0.235940, 0.165183, -0.291376, -0.178523, 0.207471, -0.245906, 0.025102, 0.082286, -0.162106, 0.348270, 0.044326, 0.162939, 0.147173, 0.131753, 0.063899, -0.117767, -0.063437, -0.040873, 0.171019, 0.219057, 0.029142, -0.075178, 0.099639, -0.504327, 0.205914, -0.343092, -0.068177, -0.118752, -0.183179, -0.313707, -0.001004, 0.070470, -0.091889, -0.035392, 0.192908, -0.297761, -0.251798, 0.060050, 0.299473, 0.186862, 0.554696, 0.044831, -0.277432, 0.005839, -0.175424, 0.238435, 0.284492, -0.083249, 2.089141, -0.059667, -0.110952, -0.088813, 0.151225, 0.004417, 0.010129, -0.253531, -0.003391, 0.089364, 0.047462, 128.035416, 0.243631, -0.009059, 0.046716, -0.080672, -0.083577, 0.042275, 0.254508, -0.015917, 0.074086, 0.214764, -0.117190, -0.159685, -0.091902, -0.006698, 0.033816, 0.313577, 0.002639, 0.402256, -0.159019, 0.284057, 0.199515, 0.055595, 0.087644, -0.001673, 0.368397, -0.188550, -0.153557, -0.021104, 0.060794, 0.062426, 0.210945, 0.122277, -0.217037, 0.047559, -0.177948, 0.273465, -0.091930, -0.462368, 0.020526, -0.216667, 0.207795, 0.006882, -0.138763, 0.122169, 0.071331, 0.081117, 0.048658, 0.260355, 0.156726, 0.253142, 0.019263, 0.390541, 0.129927, 0.188806, -0.271424, -0.110430, -0.185167, -0.141045, 0.107614, -0.196533, -0.012306, -0.228823, -0.198050, 0.085781, -0.169754, 0.117867, -0.159059, -0.116628, 0.118273, -0.136205, -0.294382, 0.234800, -0.082305, 0.634765, -0.185047, 0.359074, -0.507354, -0.333176, -0.309520, -0.039822, 0.156259, 0.109899, -0.068625, -0.040817, 0.109659, -0.270881, -0.057310, -0.044889, -0.212220, -0.135536, 0.107681, -0.146635, -0.190208, -0.218266, 0.086856, -0.367081, -33.202503, 0.003783, -0.455440, -0.035135, 0.251030, 0.112920, -0.177102, -0.152570, -0.184287, -0.258689, 0.024834, 0.333200, 0.002540, -0.305059, 0.419342, -0.250627, 0.076749, -0.315163, 0.104527, -0.100150, -0.143103, 0.088844, 0.262496, 0.059210, 0.081645, 0.264999, 0.180298, -0.201310, -0.355523, -0.018623, 0.158026, -0.215970, -0.227295, -0.129103, -0.292071, -0.101568, -0.210586, -0.083663, -0.075845, -0.056710, -0.170921, -0.573052, -0.196168, 0.160783, -0.155820, 0.211706, -0.058327, 0.229111, -0.199880, -0.338892, -0.102859, -0.072090, 0.060379, -0.060336, -0.146846, -0.094225, 0.245834, 0.142474, -0.047509, 0.694052, 0.120725, -0.212425, 0.326080, 0.225534, -0.175262, -0.094293, 0.075752, -0.034220, 0.028216, -0.056179, 0.225970, -0.115508, 0.268641, -0.167381, -0.077895, 0.019818, 0.248856, 0.136297, 0.082808, -0.207905, -0.224350, -0.036815, -0.109487, -0.162110, 0.164743, 0.170986, -0.103860, -0.161162, -0.002949, 0.329809, -0.341357, 0.320293, -0.117678, 0.040681, 0.244909, -0.102529, -0.028248, 0.248195, -0.127643, -0.392223, -0.261628, 0.042279, -0.267078, -0.403124, -0.346835, -0.126135, 0.026575, -0.108246, -0.219726, -0.159953, -0.255984, 0.092160, 0.257765, -0.254912, 0.264403, -0.126354, 0.227511, -0.164907, 0.005162, -0.044372, -0.203782, -0.039615, 0.076344, -0.114583, -0.402915, 0.146642, 0.051102, 0.231844, 0.212463, -0.051184, 0.087662, 0.045843, 0.299983, 0.070538, -0.074908, -0.075610, 0.289235, 0.043676, 0.074709, 0.014511, 0.174053, -0.097585, 0.162795, 0.117439, -0.127525, 0.317802, 0.284071, 0.344059, 0.051126, -0.307772, 0.111522, 0.057531, 0.114367, 0.325955, -0.045534, 0.094009, 0.151480, 0.057133, 0.259606, 0.106509, 0.019307, 0.128172, 0.083182, -0.056627, -0.131448, -0.031983, 0.147915, -0.117282, 0.479631, 0.096432, 0.020996, 0.108881, -0.443413, -0.235407, -0.065474, 0.013340, 0.361426, -0.253298, 0.255839, 0.265786, -0.159752, 0.033305, -0.142989, 0.247697, -0.104230, -0.061659, 0.120357, 0.351762, 0.026226, -0.168344, 0.173498, -0.740345, 0.023732, -0.042224, 0.083644, -0.021357, 0.115013, -0.136847, 0.237994, 0.098197, -0.345184, -0.150066, -0.011991, -0.337294, 0.018879, 0.095790, 0.025246, -0.122397, 0.242818, 0.048402, -0.101960, -0.221669, 0.313242, 0.294304, -0.004074, -0.241377, -0.280270, 0.182261, -0.283162, 0.272449, -0.133263, 0.138147, -0.084929, 0.358778, 0.282228, 0.208290, 0.397303, 0.149017, 0.011664, 0.091322, -0.011152, 0.054137, 0.177479, 0.059672, 0.396519, -0.097938, 0.267105, 0.088128, 0.066401, 0.322513, 0.048144, 0.143423, 0.137689, 0.301361, -0.193647, 0.305778, 0.019487, -0.068826, -0.133921, -0.116541, 0.291430, 0.040554, 0.101785, -0.155604, -0.299589, -0.190898, 0.095497, -0.107841, 0.111929, -0.276386, -0.111135, 0.050299, 0.106591, 0.015391, -0.163569, 0.160956, -0.134311, 0.237384, -0.173322, 0.054794, 0.397545, 0.016455, 0.041949, 0.048323, 0.207314, 0.080519, 0.160725, 0.014891, 0.155450, 0.366512, -0.467380, -1.586073, 0.203469, -0.138458, 0.103246, 0.045112, 0.070606, -0.274383, -0.028020, 0.181802, -0.033015, -0.019431, 0.204992, -0.190455, 0.155745, -0.012460, 0.062127, -0.135204, 0.002476, -0.425798, -0.010014, 0.180063, -0.057760, 0.116176, 0.153568, -0.035283, 0.293177, -0.136563, -0.144873, -0.130764, -0.400337, 0.037507, 0.056780, 0.348752, 0.105969, -0.357624, -0.113779, -0.083118, 0.475611, -0.105558, -0.006774, -0.380161, -0.122769, 0.095162, -0.190912, 0.253958, 0.039832, -0.679116, -0.098878, 0.083775, -0.188979, -0.046804, -0.088732, 0.180274, -0.045188, 0.337999, -0.065618, -0.277620, -0.046494, -0.400856, 0.148373, -0.537840, 0.021332, -0.209565, 0.156998, -0.057868, 0.088022, 0.081031, 0.061787, 0.110939, -0.033602, -0.071425, -0.043439, 0.058001, 0.037356, -0.245780, 0.297307, 0.293269, 0.062854, 0.268967, -0.017883, 0.227948, 0.171781, 0.130548, -0.093293, -0.116841, 0.103891, 0.172330, -0.006833, 0.222772, 0.063618, 0.122726, 0.009994, 0.149557, -0.052636, 0.100766, -0.027217, 0.112531, 0.197093, 0.297945, -0.273636, -0.066223, -0.398221, -0.126979, -0.104331, -0.234188, 0.038832, 0.079074, 0.100316, -0.115775, -0.028307, 0.063505, -0.145569, 0.015260, 0.136817, -0.060069, 0.118199, -0.028661, 0.183354, -0.176381, 0.227108, -0.375352, -0.336468, -0.265594, 0.177987, 0.192654, 0.276853, 0.281726, -0.205698, 0.114776, 0.058981, 0.268885, -0.290771, 0.056348, -0.132106, 0.004171, -0.294268, -0.047295, -0.258064, 0.068847, -0.019094, 0.128735, -0.339064, 0.150565, 0.032204, -0.107494, -0.212357, -0.392204, 0.142309, 0.166128, 0.294170, -0.094796, 0.162447, -0.194082, 0.117672, 0.112715, -0.268617, -0.203952, 0.302323, 0.043398, -0.165024, 0.258412, -0.131611, 0.053085, -0.050716, 0.068517, 0.184580, -0.003701, 0.113024, -0.158499, 0.070886, -0.122926, -0.369319, -0.083652, 0.015595, -0.287224, 0.033416, 0.265407, 0.136693, 0.147315, 0.049587, 0.061653, 0.135263, 0.085766, -0.018176, -0.046825, 0.680900, -0.290292, -0.192008, 0.104423, 0.462203, -0.230633, -0.148510, -0.290753, -0.119373, 0.149856, -0.148972, -0.221262, 0.242384, -0.385690, 0.382514, 0.102611, 0.091951, -0.017633, 0.285416, -0.438218, -0.245008, -0.372876, -0.042345, 0.307796, 0.253065, 0.224285, 0.067665, -0.276440, -0.137734, -0.476530, 0.158134, -0.051262, 0.125688, -0.097106, -0.120296, -0.084980, -0.024210, -0.293998, -0.677513, -0.174091, 0.020636, 0.104804, -0.202423, 0.044413, 0.190037, 0.127692, -0.011159, 0.049248, 0.469955, 0.084202, 0.176607, -0.083388, 0.474030, -0.097060, -0.179215, 0.051263, 0.203911, -0.016553, 0.021478, -0.011275, 0.080036, -0.354801, 0.027893, -0.376584, -1.063083, -0.184067, 0.235439, -0.053646, -0.180361, -0.067230, -0.168108, -0.087382, 0.271364, -0.409338, 0.177020, -0.079992, -0.026682, 0.159692, -0.144044, 0.084965, 0.295279, -0.016958, -0.281130, -0.124694, 0.113254, -0.092078, 0.166073, -0.201656, 0.219734, -0.002040, -0.244022, 0.189506, 0.304815, -0.104193, -0.305744, 0.420067, -0.328254, 0.184088, 0.049512, 0.387134, 0.397567, -0.088308, 0.146586, -0.034823, -0.091046, 0.374431, 0.471883, 0.287598, -0.028405, 0.249800, 0.078338, 0.346172, 0.232075, -0.469383, -0.090818, -0.080286, 0.354226, 0.045621, -0.325774, -0.247032, 0.031432, -0.119826, 0.356725, -0.186738, -0.211427, -0.080541, -0.189237, -0.292033, -0.003675, -0.022361, 0.069102, 0.128092, -0.065881, -0.018035, -0.135795, 0.292136, 0.099319, -0.320945, 0.057975, 0.086676, 0.266249, 0.363233, 0.245584, -0.171314, -0.003890, -0.169675, -0.010952, 0.091054, -0.045315, 0.050372, -0.324220, 0.311123, 0.116560, 0.021926, -0.045532, -0.233053, -0.122790, 0.274514, 0.024679, -0.120103, 0.315769, 0.077711, 0.137499, 0.145189, -0.089259, 0.496703, -0.136115, -0.143815, 0.062720, -0.374663, -0.138021, -0.079940, 0.089205, -0.197390, -0.131072, -0.053951, 0.295117, -0.122619, 0.008275, 0.055221, 2.527948, -0.072510, 0.226574, -0.000553, 0.000469, 0.058099, 0.227194, 0.057084, -0.110474, -0.039402, -2.520882, -0.173620, 0.074343, -0.078293, -0.008632, 0.207596, 0.073645, -0.251133, 0.115755, -0.080834, -0.041513, 0.484292, 0.168957, 0.118304, 0.022164, 0.035524, 0.157908, 0.042480, -0.340496, 0.024886, 0.188204, 0.062612, -0.275289, -0.018369, -0.119194, 0.317514, -0.388682, -0.052790, -0.011951, 0.218163, 0.100938, -0.280899, -0.303154, 0.014371, -0.111063, 0.059828, -0.244759, -0.016609, 0.065133, -0.492524, -0.040304, 0.368377, -0.203313, -0.122140, -0.103725, 0.224113, 0.069456, -0.093167, -0.236615, -0.130409, -0.202502, -0.021742, -0.090291, 0.225696, -0.023631, -0.117515, 0.075394, 0.091924, 0.187314, 0.030512, -0.075208, 0.352388, 0.018281, 0.150661, 0.127157, -0.151813, -0.008576, 0.272332, 0.065013, -0.057718, 0.185111, 0.269063, 0.109588, -0.237452, -0.234355, -0.101361, -0.283299, 0.218517, 0.051335, 0.140085, -0.062215, 0.458092, 0.229132, 0.172910, 0.150229, -0.053364, 0.309130, 0.009093, 0.120336, 0.160940, -0.357098, -0.067086, 0.252230, -0.117884, -0.112602, 0.044955, -0.237630, 0.070310, -0.024365, 0.156219, -0.114220, -0.091250, 0.103548, -0.375174, 0.319262, -0.249514, 0.044603, -0.280420, -0.109239, 0.174080, -0.089466, 0.247686, -0.349333, -1.858536, -0.059449, 0.059814, -0.012567, 0.213229, -0.083366, 0.166048, 0.152955, 0.364384, -0.016764, -0.377169, -0.021926, -0.280078, -0.256957, 0.219507, 0.323770, 0.064709, 3.167198, 0.133371, -0.154305, 0.155670, -0.418494, 0.186039, 0.267429, -0.052644, 0.358142, -0.026855, -0.175574, 0.072212, -0.153460, -0.001919, -0.168257, 0.376808, -0.099921, -0.187636, 0.190089, 0.186359, -0.157095, 0.053879, 0.024165, -0.071112, 0.010843, -0.120587, -0.195333, -0.328524, 0.004142, -0.111388, 0.031758, -0.053832, 0.073116, -0.051583, 0.038443, 0.105318, 0.241116, -0.213961, -0.008212, -0.162338, 0.098530, -0.298992, -0.031188, -0.221301, -0.312894, -0.039141, 0.210316, -0.067856, 0.035476, 0.445277, -0.242866, 0.041592, -0.035022, -0.234247, -0.259067, -0.258806, -0.325364, 0.234251, -0.235741, -0.313767, -0.020750, 0.014165, -0.139571, -0.146344, -0.098713, -0.199823, 0.088635, 0.075124, -0.391888, 0.062471, -0.026739, -0.385888, 0.050499, 0.106950, 0.101156, -0.033433, 0.136926, 0.355365, -0.101414, 0.246039, 0.305378, 0.524674, -0.071783, -0.042693, -0.501538, 0.298853, -0.205159, -0.183165, 0.035508, -0.072174, -0.262256, 0.173090, 0.017663, -0.428835, 0.377085, -0.198629, -0.088968, -0.152348, -0.322048, -0.200595, -0.155210, 0.225595, -0.169172, -0.069389, 0.089924, 0.142565, -0.107629, -0.121446, 0.160224, 0.066458, -0.029742, -0.056111, 0.088810, 0.068121, -0.137926, -0.133753, -0.472685, -0.257479, 0.083383, -0.146864, 0.041727, 0.073747, 0.145950, -0.138010, -0.036917, -0.015869, -0.116263, -0.185831, -0.205993, 0.132732, -0.047409, -0.006760, -0.080753, 0.278811, 0.390038, 0.055438, 0.085052, 0.359418, 0.341152, -0.001708, -0.223373, -0.116909, -0.179805, -0.224293, 0.004220, 0.147470, 0.412308, 0.179721, 0.035299, 0.093562, 0.084002, 0.059205, 0.010461, -0.077142, 0.129577, -0.064062, -0.222933, -0.377122, 0.144834, 0.043854, -0.090149, 0.644857, 0.063935, -0.240127, 0.094337, -0.348533, -0.036197, -0.092717, 0.102307, -0.214358, 0.293546, 0.239970, 0.024064, -0.037048, -0.312530, -0.043863, 0.172022, 0.280025, 0.033667, 0.392179, -0.121837, 0.198296, 0.190774, -0.148779, -0.084748, 0.203320, -0.075756, -0.272525, 0.165499, 0.047926, 0.094205, -0.150766, 0.298233, 0.122330, -0.347350, 0.093496, -0.421884, -0.021886, -0.050008, 0.209601, 0.343015, -0.206028, 0.147506, -0.021370, -0.140223, 0.084991, -0.031831, -0.298838, -0.076371, 0.047491, -0.073452, 0.060541, 0.134042, 0.138549, 0.060398, 0.461653, 0.078567, -0.090262, -0.051376, -0.199081, 0.297879, -0.088592, -0.182951, -0.262351, -0.184993, 0.276619, 0.135889, -0.046125, -0.065739, 0.326954, 0.256820, 0.004986, 0.240360, -0.148461, -0.296636, -238.705795, 0.279015, -0.227116, -0.075478, -0.049478, -0.003246, 0.160058, 0.072642, -0.213787, 0.205725, 0.040510, -0.167095, 0.284532, -0.162913, -0.110715, 0.100040, 0.072398, 0.036747, 0.144605, 0.037317, -0.077109, -0.195730, -0.160322, -0.056836, 0.027718, 0.382668, -0.018534, -0.116547, 0.120052, 0.450378, -0.051149, -0.027894, 0.082928, 0.110767, 0.238183, -0.067702, 0.274300, -0.178344, -0.013695, 0.012706, 0.633373, -0.451463, -0.168249, 0.062861, -0.019580, -0.092006, 0.168151, -0.114727, 0.086298, -0.358783, -0.104758, -0.092433, -0.269179, -0.303982, 0.341377, 0.284413, -0.082784, -0.189649, -0.324978, -0.007991, -0.100932, 0.093140, 0.121607, -0.125555, 0.041311, -0.283090, -0.180703, 0.095891, -0.104645, 0.094700, 0.028300, -0.429573, -0.038848, 0.045438, 0.296083, 0.414388, -0.231478, -0.132268, 0.499307, -0.190475, -0.091800, 0.033520, -0.392250, -0.051865, 0.046909, 0.080499, -0.111119, 0.001747, -0.024908, 1.435587, -0.116273, 0.028596, -0.219918, 0.217898, 0.109079, -0.294474, 0.188478, -0.075871, 0.045406, 0.221870, 0.134385, 0.406741, -0.136979, 0.025424, 0.310890, -0.338089, -0.267803, 0.196967, 0.064867, -0.326322, -0.003199, 0.162069, -0.539754, 0.248751, -0.402042, 0.008645, 0.340092, -0.274951, -0.149770, 0.022828, 0.309340, -0.076141, 0.202228, 0.155451, -0.049615, 0.124350, -0.187734, -1.008402, 0.043390, 0.123346, -0.043236, 0.131659, -0.023350, 0.032426, 0.171453, 0.019514, -0.027626, -0.482803, -0.111325, -0.035046, 0.077589, 0.040191, -0.412283, -0.023948, -0.255784, -0.182507, 0.174761, 0.157958, -0.000148, -0.069821, 0.287009, -0.448967, 0.016811, 0.096958, 0.017853, 0.497372, 0.166014, 0.072999, -0.217196, -0.028671, 0.012037, 0.106757, -0.069543, -0.437547, -0.388524, 0.030020, -0.309980, -0.229058, 0.378801, 0.082238, 0.376137, -0.279252, -0.176498, 0.240532, 0.610819, 0.139535, 0.269533, 0.092485, -0.136502, -0.192036, 0.050561, 0.097398, 0.372029, -0.001824, -0.301421, -0.031866, 0.239669, 0.178366, -0.189994, 0.242050, -0.077198, 0.184283, -0.160378, -0.065293, 0.022250, 0.112066, -0.237611, 0.048226, -0.411141, -0.352683, -0.175490, -0.022692, -0.246537, -0.116630, 0.361260, 0.215786, 0.045674, -0.351500, -0.262376, -0.085229, -0.219014, 0.019868, 0.368068, 0.194790, 0.089863, -0.362299, -0.211156, -0.201661, -0.136457, -0.042994, -0.423544, -0.125322, -0.086215, 0.048890, -0.086572, 0.456497, -0.209037, -0.183820, 0.251238, 0.326131, 0.163549, -0.103553, 0.003541, -0.047487, -0.219334, -0.100857, -0.119177, -0.093215, -0.591474, 0.037228, -0.089278, 0.201346, 0.125002, 0.000785, -0.124018, 0.113703, 0.001804, -0.222971, -0.074210, -0.228926, 0.113468, 0.164455, 0.229040, 0.312711, -0.159408, 0.044091, 1.640864, -0.163485, -0.076737, 0.164283, 0.131292, 0.095390, -0.150891, -0.184106, -0.070647, 0.056543, 0.024427, 0.056847, -0.108406, 0.011860, -0.161806, 0.160051, 0.171704, 0.100885, 0.124327, 0.241787, -0.126322, 0.103343, -0.033154, -0.494086, -0.068875, 0.192586, -0.112111, -0.005246, 0.073458, 0.035625, 0.345371, -0.202806, 0.057432, 0.089789, -0.048221, -0.160968, -0.346664, 0.108492, 0.284215, 0.216231, -0.159446, -0.136628, 0.419226, 0.006298, -0.084110, 0.112085, 0.193164, -0.032155, -0.367998, 0.134539, 0.222541, -0.167029, -0.275256, -0.151384, 0.053744, -0.089860, -0.060655, 0.230576, 0.206137, -0.117398, -0.114857, -0.009261, 0.109873, -0.469512, -0.362118, 0.007362, 0.021641, -0.034063, -0.251200, -0.393442, 0.012432, -0.293028, -0.173386, 0.061828, -0.916887, 0.006624, 0.088955, -0.227887, 0.311972, 0.050661, -0.199971, -0.019508, 0.020391, -0.184338, 0.098965, 0.308776, -0.148873, -0.187788, 0.084654, 0.172593, 0.127041, 0.160375, -0.378799, 0.016060, -2.151115, -0.090067, 0.047538, 0.503685, -0.039138, -0.072460, 0.316482, 0.088396, -0.110424, 0.167484, -0.179726, 0.267681, -0.192473, -0.115660, 0.068996, -0.176914, 0.028367, -0.127366, -0.007064, 0.110073, 0.002044, 0.052381, 0.066706, -0.421245, -0.181691, -0.018712, -0.095661, 0.071816, 0.133783, -0.079549, 0.283978, -0.006812, 0.065023, -0.084458, 0.036490, -0.120879, 0.185897, -0.100925, -0.237208, -0.221025, 0.112360, 0.630928, 0.193131, 0.226449, 0.168891, 0.030022, -0.234973, 0.053762, -0.034144, -0.117988, -0.150061, -0.164141, 0.105524, 0.236857, -0.199115, 0.079895, -0.360218, -0.148703, 0.260559, -0.160007, 0.147693, 0.865534, -0.143321, 0.025805, -0.148288, -0.171598, -0.251012, 0.008931, -0.273654, -0.067131, 0.075692, -0.140921, 0.091426, 0.226167, -0.128620, 0.372039, -0.157309, 0.102008, -0.076611, 0.198180, -0.076270, 0.083497, -0.075266, -0.226794, -0.056273, 0.243448, -0.022010, 0.364583, 0.175917, -0.128310, -0.108197, 0.121737, -0.521044, -0.098395, -0.226309, 0.045766, -0.341908, -0.124378, 0.277148, -0.155913, -0.266295, 0.152795, 0.143411, -0.077569, 0.109936, 0.063535, -0.186573, -0.294061, -0.227156, -0.188745, 0.002040, 0.072413, 0.191117, 0.234375, 0.188056, -0.178947, -0.130463, 0.230727, -0.354185, -0.043358, -0.144333, 0.194533, -0.033128, 0.043025, 0.012391, 0.068538, -0.120264, 0.103728, -0.049720, 0.012240, -0.228349, -0.160189, -0.076711, 0.365869, -0.128137, 0.376037, -0.375921, 0.161637, 0.405264, 0.042779, -0.195387, -0.152188, 0.161809, 0.009019, -0.047327, -0.042773, 0.056560, 0.071051, -0.114705, -0.002205, -0.125295, 0.105547, 0.223823, 0.086776, -0.178148, -0.419199, -0.036069, 0.036212, 0.081943, -0.170275, 0.218658, -0.044451, -0.382531, 0.026978, -0.149995, -0.153278, -0.218451, -0.111461, 0.030944, 0.028412, 0.013353, 0.258120, 0.255989, -0.392538, -0.173874, 0.080952, -0.080850, 0.096569, -0.389507, 0.217946, -0.083177, 0.076047, 0.037717, -0.026658, 0.300781, 0.519396, 0.430522, 0.430339, -0.346492, -0.121488, 0.011706, 0.175700, -0.057951, 0.449560, 0.170727, 0.056294, 0.137565, -0.045343, -0.250900, 0.303868, -0.268543, -0.136157, 0.216486, -0.182460, 0.227364, 0.088573, -0.021023, -0.325782, 0.418006, 0.207157, -0.151872, 0.262107, 0.187948, -0.018202, -0.340040, 0.349899, 0.189199, 0.086197, 0.407271, 0.027902, 0.040060, -0.068500, 0.110478, -0.151632, -0.079098, -0.057845, -0.209543, -0.145532, 0.137063, 0.354927, -0.338940, -0.031344, 0.006183, -0.183120, 0.016596, 0.039544, -0.165709, -0.290147, -0.165053, 0.301226, 0.392800, -0.135610, 0.396023, -0.040169, -0.231163, -0.196984, -0.055680, 0.150472, -0.052747, 0.143330, -0.227769, 0.009763, 1.472614, -0.073340, 0.155742, 0.237754, -0.188790, 0.030732, -0.331827, 0.328493, -0.303689, 0.073579, 0.155436, -0.308703, 0.038644, 0.286081, -0.236795, 0.010691, 0.026109, 0.220487, -0.260792, -0.472584, 0.143510, 0.123946, -0.120784, -0.017273, -0.037451, -0.102587, 0.151332, -0.055333, -0.274714, -0.014496, 0.256115, 0.055907, -0.102192, -0.060407, -0.143897, -0.005154, -0.331389, -0.091710, -0.334196, 0.021619, -0.139742, -0.233928, -0.048441, 0.261800, -0.150874, 0.177553, -0.073429, 0.104668, -0.185368, 0.525764, -0.139666, -0.385664, -0.158857, 0.010400, 0.148711, 0.119918, 0.285566, 0.057277, -0.099970, -0.029352, -0.055765, -0.205407, -0.085195, -0.073088, 0.028030, -0.025548, 0.007528, -0.058723, 0.188923, -0.034245, -0.085852, -0.220739, -0.295756, 0.095276, -0.334975, -0.272030, 0.062519, 0.118989, 0.096862, -0.119317, 0.560671, -0.130041, 0.231805, 0.403640, -0.046250, 0.043560, 0.285733, -0.061538, -0.063723, -0.055881, 0.046878, 0.129165, 0.091329, -0.132009, 0.062691, -0.017109, 0.260641, -0.059997, -0.193404, -0.280013, -0.109361, 0.133612, -0.169919, -0.161698, -0.100797, 0.023311, -0.180247, 0.024625, -0.133788, -0.039775, 0.272908, 0.026616, -0.273676, -0.155434, 0.056833, 0.201522, 0.169984, -0.010472, 0.102565, 0.155472, 0.038963, 0.116742, -0.193065, -0.039583, 0.057909, 0.143787, 0.184707, -0.058147, 0.254736, -0.059629, 0.404123, 0.221467, 0.086802, -0.026789, -0.107126, 0.352508, -0.280083, 0.157663, -0.072340, -0.210070, 0.173006, 0.021955, -0.272859, -0.142889, 0.185116, 0.160969, 0.058252, 0.037985, -0.187131, 0.213441, 0.546652, -0.032847, -0.414952, 0.021090, -0.244166, 0.237991, -0.081623, 0.148767, 0.309396, -0.073836, 0.098373, 0.095644, 0.158426, -0.007794, -0.139414, -0.498481, 0.230803, -0.162293, -0.050418, -0.021728, -0.383829, -0.041520, 0.305914, -0.049986, 0.263415, 0.405742, 0.191876, 0.285284, 0.096262, 0.130825, 0.055108, 0.317677, 0.077664, -0.334160, -0.341227, 0.009171, 0.111004, 0.009803, -0.045023, 0.186045, 0.126399, -0.205398, -0.070559, 0.150518, 0.372264, -0.130678, -0.213470, 0.499703, 0.305864, -0.216665, 0.250425, 0.061312, 0.104789, 0.037182, 0.065483, -0.264854, 0.031790, -0.189471, 0.158618, 0.079253, -0.075728, -0.053936, -0.371737, 0.313983, 0.072126, -0.048012, -0.197952, -0.242474, -0.385776, 0.507576, 0.007578, 0.079459, 0.018020, 0.297529, -0.236362, -0.103692, 0.071144, -0.089975, -0.186953, 0.186705, 0.021374, 0.042414, -0.101045, -0.261753, 0.070025, 0.123904, 0.547183, -0.185177, -0.196386, -0.058289, -0.129754, 0.203270, -0.337903, -0.084550, -0.116589, 0.433404, -0.127172, -0.122332, 0.308851, 0.094654, 0.098454, -0.122681, -0.021118, -0.094950, -0.123614, 0.073227, -0.709119, -0.132529, -0.078392, 0.005928, 0.280422, -0.090107, -0.241011, -0.076530, -0.049039, -0.294249, 0.241201, 0.310767, 0.162083, -0.113769, 0.228139, 0.067435, -0.035742, -0.021910, 0.064167, 0.069145, -0.281455, 0.006692, 0.116479, 0.172358, 0.160337, 0.046126, -0.241633, 0.060751, -0.005953, 0.216168, -0.167247, 0.103044, -0.158347, 0.102183, -0.167725, 0.167028, -0.063812, 0.147185, -0.212263, 0.092041, -0.824060, -0.273847, 0.067589, 0.048165, 0.103247, 0.131554, 0.429580, 0.406786, 0.044453, 0.032439, -0.091206, -0.035021, -0.464327, 0.156725, 0.148301, -0.047212, -0.052591, -0.265944, -0.069160, 0.079664, -0.390810, 0.249521, 0.158521, 0.028874, -0.010735, -0.460994, -0.086012, -0.242948, -0.099427, -0.103588, 0.352678, 0.023253, -0.129778, 0.322218, 0.285324, 0.016471, 0.395845, -0.259924, -0.099130, -0.045653, 0.196146, -0.448041, 0.154647, 0.153232, -0.002454, -0.016580, -0.270360, 0.035954, -0.184495, -0.126306, 0.230412, 0.276714, 0.332536, 0.174586, 0.121155, 0.596715, 0.030864, 0.233720, -0.499148, -0.074243, 0.230874, -0.071256, 0.091842, 0.244083, -0.392546, -0.069954, 0.304008, -0.121868, 0.206948, 0.122020, 0.416445, -0.047640, -0.069463, 0.139740, -0.114569, 0.098988, -0.014207, 0.185766, -0.070787, 0.273233, -0.024478, 0.136696, 0.151897, -0.167124, -0.226203, -0.361910, -0.022700, 0.401243, 0.294864, 0.121518, 0.131616, 0.057742, 0.012071, 0.075906, 0.336230, -0.091583, -0.171957, -0.338542, -0.393355, -0.009550, 0.082741, 0.001175, -0.162358, -0.037548, 0.220027, -0.129766, -0.233227, 0.087132, -0.352515, 0.214933, 0.137626, 0.487347, -0.180975, -0.157762, -0.027926, 0.230518, -0.098369, -0.150721, -0.098718, -0.015237, 0.235023, 0.245709, -0.261418, 0.139791, -0.206201, -0.197977, 0.008415, 0.415606, 0.099997, -0.160790, -0.155159, -0.466496, -0.010114, 0.197052, -0.004806, -0.147590, 0.104043, -0.298260, 0.188049, 0.027059, -0.905989, 0.121997, -0.310799, -0.166950, 0.007442, 0.169912, -0.135616, 0.362409, -0.196439, -0.045072, -0.332400, 0.632317, -0.120340, -0.038049, 0.228688, 0.298949, -0.087880, 0.134498, 0.044817, 0.314689, -0.047807, -0.178882, -0.141252, 0.001854, -0.038018, -0.232055, -0.074062, -0.144296, -0.176273, -0.043474, -0.125293, -0.071203, 0.148512, 0.283967, -0.209933, -0.007268, 0.256911, 0.222713, 0.093756, 0.006337, -0.060542, -0.467067, -0.345432, 0.384110, 0.084378, -0.224295, 0.220033, -0.200423, -0.047120, 0.131804, 0.117079, -0.221571, 0.101030, -0.032646, -0.099480, 0.052907, -0.180519, -0.019035, 0.011141, -0.079401, 0.003510, -0.305918, 0.154423, -0.148219, -0.245041, 0.141112, -0.033422, 0.388392, 0.287036, -0.235216, 0.063902, -0.055436, 0.154473, 0.188477, -0.044997, 0.033362, 0.351229, 0.102547, -0.050997, 0.223250, 0.166058, 0.198965, -0.097854, -0.071305, -0.076835, -0.127152, -0.161794, -0.279472, 0.064953, -0.114404, 0.211120, -0.116849, -0.206433, -0.250875, -0.416561, -0.206281, -0.084295, -0.229558, -0.068287, 0.298432, -0.107080, 0.104896, -0.065626, -0.349512, 0.111690, 0.074710, -0.015936, -0.465781, 0.023015, -0.000851, -0.112977, 0.021652, 0.046316, -0.362228, 0.110169, 0.046040, -0.124839, 0.036512, -0.059709, -0.167011, 0.040527, 0.152573, 0.278565, -0.040131, 0.075570, -0.142031, 0.064986, 0.089573, -0.152573, -0.099273, -0.300745, 0.042042, 0.129326, -0.223118, -0.432981, -0.231665, 0.138286, -0.299498, -0.007091, -0.002522, -0.042010, -0.089950, 0.169680, -0.112436, 0.145602, -0.036011, -0.139941, 0.069518, -0.002130, 0.069993, 0.215958, 0.109215, -0.151038, -0.107376, 0.062824, -0.009224, 0.177981, -0.199332, 0.115359, 0.002098, -0.464907, -0.043310, 0.269114, 0.056349, 0.096060, 0.127373, 0.088242, -0.153574, 0.025003, -0.292835, 0.125097, 0.057609, -0.118159, 0.109902, -0.033631, -0.005885, -0.235323, -0.057122, 0.393253, -0.271844, 0.161666, 0.353789, 0.035729, 0.057463, -0.114084, -0.197859, -0.058904, -0.219049, 0.170893, 0.072779, 0.082536, 0.118064, 0.127045, -0.037568, -0.034509, 0.154779, -0.295909, 0.216559, 0.233419, 0.228182, 0.035807, 0.294817, -0.129304, -0.128393, 0.115915, 0.291397, 0.093980, 0.461367, -0.141117, -0.345259, -0.155644, 0.094998, 0.143420, -0.155440, 0.163045, 0.286251, 0.121402, -0.121756, -0.035068, 0.128450, -0.109638, 0.070940, 0.053334, -0.286988, 0.102754, -0.439167, 0.024396, 0.040894, 0.071758, -0.283797, -0.018879, 0.118093, 0.105655, 0.109290, 0.283767, -0.104839, -0.915190, -0.046473, -0.035785, 0.208600, 0.174186, -0.046286, 0.288322, -0.247121, 0.037623, 0.069812, -0.100539, -0.003253, -0.079773, 0.102938, 0.218375, -0.163961, -0.092704, 0.035276, -0.046268, 0.266264, 0.222437, -0.250356, 0.289091, 0.001808, 0.047329, 0.472058, -0.172736, -0.130332, 0.286500, -0.227638, -0.080091, -0.043982, 0.250013, 0.248046, 0.015482, -0.393312, -0.081104, 0.448932, 0.279728, 0.087169, -0.224928, -0.360825, -0.286112, 0.151291, 0.215553, 0.014399, -0.244032, 0.300338, -0.131355, 0.078483, -0.315427, -0.213509, 0.095424, -0.204684, -0.134618, -0.030842, -0.139851, -0.143079, 0.099183, -0.370192, 0.098330, 0.028571, -0.212862, -0.158318, 0.316602, 0.194853, 0.101022, 0.034754, -0.189449, -0.279371, -0.212051, -0.113043, 0.154550, -0.198390, 0.223687, -0.038535, -0.241375, -0.048202, 0.175800, 0.039131, 0.320138, 0.194651, -0.105665, 0.007892, -0.298054, -0.038901, 1.050818, -0.287893, 0.033782, -0.092414, -0.028556, -0.281832, -0.085192, 0.104480, -0.485409, 0.032023, 0.109034, 0.087413, -0.005370, 0.263887, -0.166009, 0.007772, -1.354689, 0.270887, -0.051926, 0.061335, 0.018344, 0.347931, 0.162826, 0.209718, 0.114306, -0.281166, -0.045904, -0.089149, 0.148793, 0.025473, 0.084521, -0.201605, -0.318991, 0.214578, -0.037794, -0.075982, -0.059598, 0.299201, -0.305224, 0.188492, 0.019561, -0.003773, 0.272812, 0.007852, -0.177909, -0.063497, 0.151214, -0.422440, 0.421259, 0.170742, 0.150690, -0.167165, -0.084585, -0.102008, 0.013217, -0.147341, -0.046574, -0.063314, 0.084868, -0.380349, -0.233978, 0.085547, 0.011975, 0.236657, -0.019690, 0.267081, 0.521068, -0.089793, 0.036281, -0.569551, 0.051381, -0.220948, 0.352372, 0.201300, -0.102823, 0.011176, -0.218582, 0.295943, -0.633942, 0.164097, 0.294015, 0.184877, -0.228102, -0.178158, -0.017998, 0.000706, -0.064521, 0.051502, -0.379477, -0.076718, -0.240841, 0.069120, -0.354960, -0.293350, -0.056624, -0.062063, 0.043639, -0.273948, -0.298395, -0.138360, 0.211717, 0.701265, -0.230972, -0.413621, -0.346696, -0.075838, -0.083173, 0.005093, -0.093237, 0.080748, 0.168426, 0.059740, -0.209602, -0.026372, 0.142364, 0.001942, -0.201685, -0.189021, 0.027367, 0.144475, -0.186461, -0.036163, -0.100290, 0.144946, -0.143537, 0.055748, 0.165507, -0.098855, -0.138435, -0.180230, 0.246634, -0.056439, 0.122972, -0.061661, -0.218960, -0.314999, 0.008152, -0.046900, -0.086662, 0.245503, -0.103636, -0.042046, 0.011182, 0.059346, -0.062746, 0.033032, -0.077377, 0.034610, -0.028047, -0.155032, 0.211335, -0.050602, -0.342043, -0.256510, 0.196297, -0.051531, -0.085107, -0.185316, 0.085057, 0.148308, 2.109797, 0.061938, 0.302314, -0.161459, 0.162948, -0.048650, 0.098442, -0.066528, -0.085827, -0.037065, -0.387574, -0.081248, 0.152722, -0.002649, 0.018099, 0.274471, 0.246455, -0.212773, 0.027912, -0.129523, 0.028914, 0.371400, 0.340225, -0.624804, 0.199541, -0.005809, 0.155528, 0.113098, -0.185405, 0.221409, -0.132160, -0.088569, -0.184942, -0.228357, -0.055149, 0.205094, 0.222713, -0.021291, 0.006523, -0.269631, -0.137739, 0.160536, 0.170055, -0.170910, -0.085112, 0.026253, -0.244464, 0.068610, 0.044967, 0.133165, 0.051220, -0.092339, 0.013917, 0.145891, -0.078450, -0.101472, -0.038392, 0.046053, 0.121058, -0.070784, -0.148233, 0.086958, -0.153995, -0.052332, 0.073340, 0.065673, 0.063829, 0.046075, -0.039880, 0.003215, 0.073366, 0.033555, 0.024356, -0.128940, -0.037760, -0.098422, -0.009670, 0.145473, -0.106517, -0.131222, 0.033297, -0.049149, -0.081665, 0.128538, 0.196236, -0.080342, -0.012141, -0.112823, 0.071420, 0.017868, -0.002456, -0.067333, 0.037623, 0.107441, 0.022360, -0.079734, -0.120306, 0.042631, -0.105765, 0.013004, 0.017308, 0.111655, -0.105092, 0.097670, 0.074254, 0.123343, -0.000029, 0.247862, -0.077392, -0.095211, 0.147482, -0.065335, 0.200351, 0.043848, 0.028782, 0.096165, 0.035625, -0.177140, 0.063965, 0.238265, 0.118634, -0.114873, -0.125525, -0.013900, 0.136230, 0.063671, -0.245850, 0.063212, 0.055024, -0.145902, 0.050994, 0.055361, 0.317387, -0.014252, 0.031719, -0.073501, -0.014346, 0.062877, -0.041314, 0.040513, -0.205436, 0.141859, -0.029698, -0.310212, -0.049965, 0.167827, -0.034408, 0.195449, -0.080761, -0.055519, -0.089496, -0.110481, 0.131292, -0.086676, 0.081896, -0.066262, -0.309079, 0.125769, -0.004469, 0.069239, 0.132949, -0.074464, -0.078727, 0.089676, 0.045531, 0.053205, -0.234463, -0.190208, -0.199137, 0.165049, -0.054126, -0.004020, -0.015882, -0.168255, -0.068956, 0.157096, 0.098650, 0.088951, 0.081103, -0.005485, 0.052773, 0.130083, 0.101616, -0.059037, 0.133718, 0.025448, 0.048866, 0.036762, 0.084643, -0.009386, -0.032293, 0.046734, -0.259648, 0.210898, 0.079287, -0.114035, 0.048225, -0.145238, -0.028507, -0.073742, -0.093280, 0.047754, -0.170102, -0.145146, 0.015547, 0.058626, 0.125388, -0.043208, 0.025296, -0.093866, 0.038254, -0.105191, 0.189420, 0.035581, -0.162749, 0.024272, -0.225733, 0.023414, -0.030696, -0.124724, -0.133898, -0.117076, -0.005871, 0.195472, -0.031929, -0.030172, 0.135357, -0.081132, 0.156708, -0.128093, 0.097957, -0.137871, 0.000064, 0.029718, 0.155340, 0.223269, -0.061614, -0.462642, 0.047641, 0.106455, 0.011151, -0.128854, 0.046557, -0.103515, 0.035822, 0.307952, -0.165491, -0.003790, -0.015647, -0.042538, -0.021318, -0.128304, 0.215623, -0.013245, -0.034964, 0.074222, 0.041957, -0.076970, -0.031201, -0.054535, -0.033490, 0.106345, -0.141108, 0.212346, -0.061244, 0.103579, 0.129600, 0.169684, 0.207624, -0.005983, 0.120372, -0.271393, 0.109959, -0.057102, -0.036773, 0.040831, -0.083654, -0.047190, -0.067820, 0.289107, -0.029533, 0.027713, 0.185194, -0.124393, -0.028716, -0.052039, 0.022056, 0.145267, -0.043584, 0.064813, 0.013849, -0.205942, -0.039239, 0.062557, -0.102156, 0.037509, -0.031952, 0.028491, 0.171640, 0.006214, 0.457414, 0.136422, 0.086825, -0.064584, 0.183061, 0.040063, 0.093959, 0.019610, -0.059165, -0.044034, 0.198248, 0.055020, -0.083361, -0.182300, -0.005534, 0.133895, 0.143344, 0.045151, 0.091716, -0.081953, 0.142692, -0.016078, 0.068200, 0.060609, -0.124682, 0.121267, 0.015777, 0.012340, -0.159060, 0.044262, 0.179046, 0.085702, 0.001191, 0.045733, -0.026883, 0.260643, -0.064803, -0.114704, 0.004581, -0.090268, 0.035379, -0.130874, -0.000536, 0.192869, -0.163799, 0.028662, -0.272944, -0.012526, 0.052394, 0.075651, -0.005619, 0.134431, 0.023510, 0.028505, 0.164824, -0.080067, 0.111380, 0.020466, -0.108426, 0.339943, -0.085333, 0.126549, 0.101226, -0.014091, 0.011142, -0.049579, 0.097802, -0.064725, 0.069266, -0.012943, 0.107089, -0.002984, -0.207711, -0.088608, -0.097620, -0.098614, -0.007091, 0.182327, 0.145023, 0.087743, 0.054934, 0.109205, -0.405268, -0.015598, -0.173865, 0.041987, 0.159434, -0.025343, -0.011561, 0.066108, 0.106778, -0.114773, -0.141224, -0.181139, -0.211020, -0.082901, -0.088477, -0.018615, 0.193872, 0.093038, -0.022106, -0.112455, -0.149319, -0.167382, -0.007957, -0.143033, -0.495180, 0.064092, 0.133269, -0.108337, -0.041018, -0.063205, -0.054982, -0.106400, 0.094740, -0.150964, -0.052158, -0.109440, -0.165999, -0.070576, -0.017768, 0.022206, -0.043063, 0.131733, 0.103763, 0.011656, 0.194656, 0.198212, -0.129449, -0.117510, -0.051068, -0.015313, -0.037086, -0.138871, 0.164388, 0.059892, 0.005011, -0.059801, 0.082275, -0.101717, -0.015836, -0.009928, 0.259915, -0.042485, 0.103498, 0.028015, -0.086707, -0.075110, 0.021419, -0.099963, 0.050967, 0.168854, -0.112660, 0.014811, 0.013393, 0.022405, -0.100321, 0.078379, -0.062255, 0.149374, 0.110477, -0.023903, -0.108872, -0.049453, -0.052337, 0.002334, 0.024745, -0.088640, -0.187264, -0.142128, -0.021884, 0.100998, -0.055919, -0.139303, 0.116364, 0.171947, 0.050774, 0.151893, -0.145935, 0.055197, 0.090793, -0.057924, -0.048530, 0.045412, 0.067959, -0.071586, -0.018396, 0.018313, 0.152526, -0.101152, -0.011929, 0.130467, -0.044667, -0.216596, -0.156597, -0.115514, -0.052866, 0.143182, -0.081491, -0.056916, -0.187875, -0.014781, 0.197079, 0.003931, -0.157468, -0.315794, -0.055949, -0.046850, 0.146868, -0.046134, -0.036199, -0.046704, 0.221937, 0.012526, -0.174871, -0.078405, 0.066137, 0.034634, -0.019324, -0.044946, 0.053082, 0.096114, 0.060222, 0.101949, -0.109034, 0.019921, -0.129401, 0.011047, 0.169141, -0.006720, -0.111801, 0.006210, -0.143140, -0.564108, 0.039868, 0.095770, 0.189925, -0.101212, 0.122395, 0.021050, 0.057166, -0.155323, -0.144099, -0.089882, 0.109779, 0.057318, -0.072695, 0.109264, -0.027420, -0.046647, 0.003791, -0.035266, -0.036904, 0.006339, -0.174261, -0.019106, 0.131000, 0.080217, -0.016139, -0.126617, -0.175103, -0.085360, -0.105456, -0.203157, -0.067072, 0.181993, 0.040369, 0.001046, -0.240560, -0.026164, -0.098294, -0.060994, -0.097950, 0.179310, -0.009061, 0.121163, -0.243640, -0.006034, -0.036777, -0.115679, 0.015952, 0.091924, 0.024713, 0.018605, 0.153402, 0.056261, -0.032405, 0.001616, -0.139402, -0.029285, -0.039111, 0.013746, 0.002817, -0.103531, 0.013944, -0.003444, 0.041188, 0.074688, 0.159990, 0.108907, -0.010620, -0.105622, -0.038328, 0.303527, 0.010193, 0.000264, 0.054669, -0.228208, -0.008917, -0.003749, -0.086763, 0.146654, 0.192802, 0.228326, 0.178640, -0.000530, -0.073162, -0.134004, 0.408563, 0.055946, 0.050768, 0.054340, -0.004225, 0.099372, -0.099133, -0.118181, -0.017608, -0.074203, -0.202784, 0.082119, -0.131221, -0.260251, -0.049598, -0.027219, 0.098016, 0.104010, -0.212526, -0.071092, -0.117849, 0.117051, 0.202507, -0.082334, 0.178939, -0.102518, -0.032555, 0.060918, -0.054020, -0.177285, 0.019732, -0.086138, -0.187831, -0.044704, 0.168935, -0.028627, 0.215983, -0.046924, 0.010748, -0.003017, -0.041054, -0.010603, -0.209275, -0.001777, -0.102030, -0.003696, 0.007135, 0.120447, 0.089493, 0.153556, 0.008405, 0.029288, -0.004647, -0.013982, -0.063546, 0.027784, 0.104523, -0.038215, 0.034567, -0.030411, -0.190235, 0.077724, 0.077323, 0.191362, -0.023523, 0.079280, -0.081254, 0.075402, 0.128720, -0.073556, -0.193101, 0.043789, 0.048801, -0.082283, 0.114221, -0.020868, 0.031448, 0.008407, 0.233529, -0.091385, 0.043670, 0.052831, -0.015631, -0.203743, 0.011043, -0.014951, -0.109927, -0.068439, 0.068656, 0.004031, -0.132279, -0.064129, 0.021165, 0.211958, 0.209327, -0.089815, 0.092255, -0.085560, 0.032771, 0.100367, -0.018660, -0.065038, -0.029862, 0.071440, -0.015719, -0.048413, 0.010587, 0.096421, -0.068721, -0.075808, -0.117187, 0.025544, -0.104155, -0.027040, 0.063505, -0.090027, -0.023551, -0.261618, -0.031783, -0.060395, -0.073006, 0.068283, -0.043222, -0.027722, 0.190471, 0.137074, 0.140176, -0.014882, -0.042035, 0.096463, -0.144530, -0.148248, 0.016292, -0.018482, -0.031354, -0.057988, 0.056522, 0.002706, 0.139829, 0.067092, 0.003378, -0.072707, 0.049427, -0.052711, -0.024220, -0.017204, -0.039649, -0.018400, 0.115041, 0.065946, 0.091222, -0.104236, 0.004749, 0.275863, 0.076740, -0.219176, -0.189569, 0.018264, 0.185928, 0.101641, 0.041736, -0.038575, -0.016081, 0.080125, 0.157634, 0.015561, 0.000921, -0.062397, -0.035431, 0.058653, 0.054222, 0.128962, 0.150626, -0.042662, -0.172848, -0.086394, 0.126775, -0.020036, -0.034028, 0.036868, -0.055683, -0.191849, -0.165691, 0.154584, -0.042306, 0.149516, -0.140710, 0.050815, 0.149910, -0.098335, 0.021284, 0.011006, -0.025497, 0.055491, -0.175707, 0.123325, 0.037600, -0.111148, -0.144249, 0.026005, -0.067071, -0.087316, -0.104656, 0.110647, -0.048960, 0.057708, 0.068324, 0.055227, 0.088460, -0.119198, -0.147988, -0.104817, 0.275741, -0.483894, -0.087236, -0.128821, -0.008714, -0.021899, -0.029599, -0.104175, -0.115450, 0.086321, -0.113494, -0.058548, 0.105994, 0.002962, -0.325419, 0.031443, 0.217215, -0.027844, -0.096161, 0.057971, -0.093891, 0.023893, -0.043754, 0.112182, -0.091910, -0.126939, 0.033008, -0.033287, -0.100976, -0.071640, 0.027577, -0.086871, 0.001180, 0.004351, -0.130469, -0.122286, 0.158806, -0.056604, -0.101192, -0.168003, 0.120922, 0.106974, 0.009984, -0.027414, -0.145022, -0.070848, -0.015955, 0.097657, 0.235528, 0.108437, -0.050175, 0.207104, 0.161052, 0.020317, -0.033456, 0.091795, 0.150087, 0.085993, -0.066884, 0.015600, 0.019209, 0.120058, 0.010183, 0.131136, -0.220182, -0.088077, 0.194122, -0.033912, 0.072614, -0.010762, 0.082084, -0.146654, 0.014638, 0.006886, 0.112048, -0.053026, 0.032981, -0.078480, 0.002806, 0.156136, -0.004529, -0.046684, -0.008379, 0.010197, 0.018635, 0.158504, -0.010540, 0.026693, -0.018410, -0.146090, -0.223884, 0.103441, 0.169751, 0.067453, -0.072488, -0.034088, 0.129826, -0.041110, 0.033394, -0.081095, -0.023641, -0.094083, -0.131815, -0.065704, 0.106293, 0.202805, -0.186348, -0.094536, -0.123777, -0.166929, 0.152310, 0.063509, -0.033048, 0.181459, 0.173695, -0.145265, -0.029336, -0.010527, -0.038725, 0.136170, -0.067768, -0.010974, 0.067390, -0.105343, 0.059320, -0.043352, 0.111687, -0.005509, 0.012009, 0.070251, 0.077804, -0.012876, 0.088462, 0.082514, -0.029284, 0.056093, 0.004991, 0.210184, 0.140247, -0.159520, 0.201244, 0.086444, 0.076798, 0.041770, -0.037268, 0.064724, 0.016901, 0.027447, -0.119072, 0.033523, -0.129408, 0.006163, 0.045359, -0.069344, 0.154287, 0.014540, 0.048753, 0.158261, -0.239371, -0.170287, -0.032538, -0.014974, -0.035643, -0.116558, 0.126509, 0.013892, 0.080516, 0.168970, -0.093718, 0.018019, -0.151229, -0.182194, -0.169994, -0.092262, -0.072407, -0.035603, -0.017675, 0.309473, -0.009307, -0.002307, -0.043444, 0.034288, 0.012387, -0.146209, 0.031901, -0.097725, 0.091767, -0.020414, 0.127881, 0.006399, -0.164795, 0.123440, -0.109193, 0.047252, 0.140526, 0.176162, -0.135487, -0.010255, 0.091468, 0.088317, -0.024890, -0.019342, 0.038169, 0.047077, -0.055166, -0.045143, -0.145598, 0.083924, -0.201606, -0.044907, 0.298401, -0.002833, 0.046732, 0.106389, 0.161100, -0.052590, -0.044196, -0.002213, 0.097828, -0.030154, 0.066355, 0.050750, 0.075279, 0.117698, 0.040360, -0.030972, 0.005590, 0.070886, -0.030218, 0.008283, -0.150665, -0.115012, 0.055776, -0.156281, 0.020012, 0.000955, -0.012463, -0.011126, 0.113088, -0.126523, 0.072546, -0.015206, -0.012691, -0.255328, -0.020368, 0.026340, 0.045472, -0.076005, 0.020372, 0.101137, 0.016044, 0.019999, 0.082488, -0.044582, -0.142237, -0.202628, -0.110570, -0.099352, -0.094660, -0.017656, 0.116481, -0.076864, -0.009462, 0.059733, 0.238838, 0.020561, -0.038490, 0.073910, 0.115310, 0.092193, -0.110346, 0.170520, 0.095599, 0.061367, 0.155965, 0.019651, 0.085194, -0.181930, 0.052575, -0.174011, -0.010131, 0.076804, 0.030946, -0.006445, 0.172768, -0.099484, -0.105203, -0.062080, -0.143624, 2.556582, -0.167404, -0.102961, -0.139579, 0.123985, -0.208806, 0.169233, 0.006923, -0.064826, 0.046849, 0.079352, -0.056665, -0.024670, 0.029359, 0.056375, -0.100338, -0.091421, 0.050241, -0.093674, 0.098951, -0.058166, -0.137569, -0.016090, -0.066455, -0.069770, -0.007588, 0.000083, 0.120106, -0.054036, 0.253306, 0.004116, 0.038208, 0.061935, 0.170908, -0.070140, 0.030163, -0.027210, 0.006544, -0.066824, -0.175104, 0.076057, 0.190138, 0.164863, -0.214544, -0.085626, -0.010836, -0.135349, -0.090286, 0.028399, -0.013920, -0.166128, 0.036808, 0.105577, 0.009711, 0.174213, 0.093496, -0.027271, -0.048224, 0.009151, 0.037170, -0.056318, -0.088480, -0.008601, -0.089990, 0.049341, 0.066079, 0.064526, 0.016893, -0.036738, 0.015754, -0.030240, -0.005426, -0.031182, -0.044684, -0.009824, 0.004057, -0.066616, -0.090872, -0.040824, 0.168969, 0.021035, -0.070932, -0.142669, 0.074642, 0.099989, -0.056929, 0.043640, 0.036803, -0.047856, -0.223779, -0.035164, 0.065010, 0.085461, 0.062063, 0.101330, 0.025830, 0.063654, -0.122982, 0.033383, 0.041732, 0.057830, -0.078600, -0.126750, 0.179931, -0.086182, -0.074227, -0.038279, -0.021566, 0.012050, 0.085861, -0.059773, -0.023367, -0.106824, -0.002070, 0.002285, 0.212473, -0.088806, -0.078502, 0.073805, 0.158722, 0.083940, -0.006856, -0.079596, 0.019230, -0.000111, -0.031002, -0.133267, 0.034016, 0.144957, -0.070788, 0.039332, -0.051568, 0.220076, -0.048517, -0.084204, -0.053318, 0.006731, 0.026424, -0.187312, 0.122348, 0.054411, -0.091675, -0.160171, -0.018644, 0.101943, 0.018161, -0.131879, -0.191510, 0.043427, -0.103110, -0.099143, -0.045929, 0.072876, 0.046478, 0.004056, -0.163045, 0.060215, -0.086532, 0.257672, -0.062916, 0.102341, -0.047209, -0.078031, -0.041156, -0.182353, -0.010538, -0.087959, 0.032617, -0.141942, -0.052926, 0.103581, 0.007132, -0.119888, 0.057840, -0.028673, 0.073597, -0.085298, 0.070494, 0.118783, 0.140100, 0.135047, -0.117422, 0.158509, 0.052836, 0.078841, 0.163435, 0.019444, -0.073919, 0.091381, 0.310486, -0.081429, 0.102164, -0.042007, 0.201885, -0.063307, -0.021332, -0.022787, 0.109687, -0.119101, 0.040771, 0.047213, -0.066111, -0.121469, -0.194740, -0.024071, -0.124407, -0.006824, 0.080048, 0.079942, -0.019436, -0.059773, 0.017773, 0.163014, 0.004403, 0.003088, 0.073203, 0.204920, 0.179437, 0.035218, -0.139184, -0.091920, -0.108471, 0.028626, -0.133551, -0.123885, 0.081876, -0.062701, 0.166134, 0.063536, -0.130976, 0.029906, -0.043384, -0.086553, 0.058500, 0.087557, -0.054553, 0.026685, 0.019368, -0.042432, 0.035095, 0.021697, 0.028026, 0.150093, 0.014492, -0.004425, 0.071864, 0.040344, -0.124920, 0.108537, -0.017494, 0.123815, -0.105183, 0.051713, -0.007619, -0.132583, -0.060853, -0.002809, -0.087576, 0.119988, 0.209185, 0.024450, 0.280106, 0.334632, -0.084352, 0.051591, -0.206016, -0.026690, 0.113289, -0.034386, -0.072417, 0.045074, 0.172522, 0.146493, 0.013920, 0.119007, 0.058098, -0.023524, -0.079473, 0.056404, -0.011484, 0.123527, -0.089267, 0.056770, -0.140024, -0.416932, -0.005624, -0.030032, -0.172040, 0.109205, 0.050509, -0.081695, 0.076047, -0.204723, 0.089331, -0.094327, -0.112864, 0.042657, 0.135507, 0.235649, -0.250964, 0.096386, 0.036718, -0.045074, -0.030887, 0.147234, -0.155011, 0.210026, 0.011696, -0.060620, -0.134698, 0.016560, -0.104805, 0.059295, -0.109602, 0.133254, 0.017912, -0.144454, 0.013940, 0.121052, -0.026092, 0.126017, 0.059352, 0.219667, -0.198222, 0.009049, -0.103872, 0.072315, -0.059877, 0.176279, -0.157712, -0.293676, -0.089767, 0.052268, 0.252985, 0.054651, 0.014295, -0.048404, 0.094845, 0.169522, 9.682631, 0.002156, 0.029371, -0.129480, 0.072393, -0.134540, 0.088301, 0.052060, -0.075624, 0.134890, 0.010965, 0.170918, 0.135516, -0.101814, -0.021252, 0.228217, 0.146621, 0.202230, 0.007205, -0.034267, -0.050499, 0.096961, 0.226824, -0.008454, 0.000691, 0.034420, -0.111146, -0.025615, 0.047113, -0.090860, -0.075300, -0.133720, -0.069623, 0.014443, 0.080524, 0.114228, -0.120446, -0.043915, -0.068701, 0.010278, -0.046676, -0.142222, -0.128781, 0.058228, 0.001182, -0.000170, 0.019747, 0.060858, -0.002046, -0.148755, 0.113579, -0.082474, 0.185375, -0.111596, 0.041603, 0.057255, 0.129451, -0.053933, 0.018419, 0.247813, 0.108408, 0.134152, 0.089034, -0.052033, 0.006358, 0.008467, 0.005146, 0.063742, -0.091051, -0.168487, 0.199703, -0.059146, -0.236000, -0.076922, 0.175605, -0.014691, -0.085533, -0.044592, 0.005464, 0.016823, 0.011917, 0.077121, 0.089321, 0.075612, 0.054292, 0.066355, 0.075667, -0.163103, 0.048358, 0.118761, 0.020918, 0.377016, 0.003515, -0.071894, -0.057181, 0.049352, -0.066233, -2.629804, 0.157970, 0.113020, -0.033955, -0.007806, -0.099888, -0.076582, 0.036080, -0.195791, -0.117589, -0.069426, 0.109494, -0.131452, 0.053073, -0.000092, 0.060154, -0.077843, -0.019798, -0.040024, 0.020924, -0.085511, -0.056894, 0.064134, -0.272992, -0.053522, -0.127036, 0.122582, 0.038638, -0.045516, 0.135655, -0.108609, -0.074476, -0.126321, 0.110561, -0.110685, -0.047001, 0.044483, 0.070647, 0.178560, -0.096094, 0.210525, -0.028995, -0.023603, 0.017151, -0.158341, 0.110496, 0.048686, -0.053115, 0.001444, 0.017324, -0.132085, -0.048962, 0.059589, 0.120297, 0.193743, -0.063182, 0.104817, -0.145373, -0.075888, 0.020757, -0.205747, 0.246694, -0.037492, 0.089005, -0.104378, -0.013613, 0.000241, 0.067502, -0.120606, 0.215278, -0.179304, 0.012581, -0.107120, -0.132678, -0.011366, -0.022287, 0.008912, -0.035235, -0.088217, 0.129142, 0.187259, 0.043978, -0.183765, -0.048143, 0.047426, -0.059667, 0.059579, 0.134233, -0.040315, 0.050507, 0.101740, -0.029056, 0.199967, 0.076188, -0.140750, 0.118674, 0.132636, -0.029691, 0.003015, 0.049809, 0.029617, 0.095592, 0.095798, -0.034030, 0.080003, -0.101252, 0.073631, -0.007474, -0.038949, -0.129607, -0.071322, 0.114184, 0.009439, -0.100098, 0.118667, 0.184906, 0.059067, 0.023703, 0.120683, -0.215075, -0.113449, 0.031553, -0.109226, -0.195216, -0.084599, 0.062513, 0.138089, 0.049118, 0.102818, 0.120826, 0.081246, -0.038102, -0.174262, 0.093702, 0.025366, 0.117648, 0.033300, -0.093198, 0.060734, 0.133106, -0.110440, -0.001461, -0.084107, -0.028687, -0.149525, -0.006767, -0.097862, 0.049398, -0.013509, -0.050115, 0.003395, 0.084197, 0.194887, 0.172127, 0.017510, 0.003000, 0.051860, -0.041028, -0.001238, -0.097744, 0.040519, -0.070691, -0.239162, 0.039134, 0.033499, 0.030882, -0.137624, -0.031445, 0.007759, -0.001294, 0.177023, -0.225731, -0.047259, -0.028830, -0.049789, -0.162259, 0.056248, 0.089611, -0.078880, -0.078663, -0.029779, -0.072637, -0.021299, -0.018515, 0.101256, 0.009083, -0.114091, 0.032450, 0.131570, 0.080799, 0.087322, 0.026310, -0.225950, 0.223040, 0.027918, -0.113781, 0.260055, -0.041889, 0.128187, 0.073725, -0.068058, -0.095086, -0.001833, 0.082288, -0.277317, -0.136214, 0.074482, 0.049297, 0.118687, 0.103756, -0.202483, 0.173660, -0.003829, 0.070474, -0.189622, -0.119131, -0.091286, -0.107538, 0.017797, 0.017533, 0.033681, -0.001181, -0.092214, 0.164601, -0.011865, 0.071391, -0.101871, 0.020174, 0.131426, 0.032408, -0.027802, 0.064936, 0.051341, -0.028230, 0.077680, -0.075032, 0.280397, 0.131985, 0.223232, -0.071390, -0.154668, 0.087800, 0.185809, -0.018930, -0.168750, -0.006115, -0.031583, 0.187107, 0.061579, -0.040498, -0.172736, -0.006253, 0.001220, 0.073617, 0.122552, 0.029971, -0.200962, 0.050682, 0.003359, -0.154542, 0.138223, 0.034446, -0.034388, 0.047778, -0.037662, -0.019845, -0.024081, -0.068401, -0.100119, -0.034348, 0.065549, 0.012776, 0.064193, -0.004995, -0.152623, 0.048524, -0.009270, -0.001469, 0.105443, -0.014044, -0.001488, -0.239352, 0.004155, 0.123335, -0.043821, -0.131792, 0.129444, 0.041997, 0.099550, 0.042082, -0.012853, -0.050401, -0.007183, -0.084680, -0.109568, 0.038398, 0.087962, -0.056577, 0.067483, 0.105702, -0.095871, -0.067368, -0.010135, -0.009474, -0.302795, -0.058894, -0.097177, 0.026692, 0.162147, -0.010134, -0.025465, 0.074236, 0.189893, 0.208027, -0.166527, -0.080902, 0.163825, 0.075455, -0.086193, 0.067279, -0.040764, 0.095959, -0.204203, 0.225899, -0.017613, -0.082522, 0.025204, -0.003732, 0.000776, -0.128504, 0.027494, 0.048072, -0.142473, 0.043339, -0.047390, -0.053968, 0.106853, 0.043941, -0.083610, 0.082239, -0.187570, 0.031677, 0.084901, 0.000946, -0.009740, 0.214457, 0.130787, -0.023539, -0.102433, 0.037433, -0.026917, 0.209028, 0.105405, -0.082488, -0.144510, 0.017998, 0.031503, 0.083860, 0.114184, 0.062783, -0.002152, -0.068874, -0.032445, -0.013950, -0.090913, 0.112378, -0.170818, -0.012935, 0.137448, 0.112307, -0.009934, -0.129617, -0.168887, 0.093721, -0.019550, -0.171234, -0.170375, -0.104301, -0.031343, 0.193374, 0.004884, -0.085442, -0.116577, 0.078748, 0.056595, -0.123327, -0.051144, 0.134382, -0.010035, -0.232141, -0.018366, 0.006285, -0.114391, -0.227461, -0.225165, 0.067436, -0.094171, 0.046118, 0.019570, 0.030311, 0.047347, -0.077433, 0.136164, -0.267984, -0.039637, 0.021371, -0.026968, 0.069256, -0.016582, -0.066447, 0.031114, 0.065368, -0.182327, 0.093436, -0.009730, 0.369502, 0.016920, 0.131156, -0.131906, -0.087866, 0.080246, 0.208192, -0.024781, -0.033803, -0.231204, 0.021936, -0.023622, 0.065268, -0.031379, 0.300945, -0.202644, 0.111867, 0.042472, 0.132615, 0.070704, 0.054136, -0.069640, -0.042350, 0.122192, -0.212620, 0.056583, 0.142031, 0.182399, 0.045504, -0.026424, -0.161505, 0.120686, 0.008322, -0.026457, 0.023266, 0.076771, -0.085146, -0.088130, 0.066825, 0.031470, -0.026106, -0.079418, -0.032650, 0.001467, -0.014045, 0.040232, -0.014424, 0.032059, -0.120469, -0.082038, 0.063904, 0.154969, 0.043723, 0.034862, -0.065451, -0.143833, -0.237235, 0.036208, -0.046621, -0.048227, 0.005453, -0.025634, -0.226653, -0.090916, 0.087490, 0.009247, 0.160890, 0.032906, -0.012335, 0.129576, 0.206885, 0.097189, -0.144431, 0.098787, -0.229132, 0.265071, -0.081921, 0.118975, 0.080841, 0.011746, -0.144160, 0.162737, -0.052150, -0.205903, -0.095993, -0.096103, -0.019133, -0.039946, -0.111494, -0.003226, -0.037728, 0.087698, 0.043699, 0.052581, 0.001568, 0.145757, -0.132908, 0.001313, -0.064447, -0.044667, 0.062395, 0.042453, -0.106639, -0.135996, -0.034514, -0.021694, -0.019063, 0.127528, 0.215833, -0.015655, 0.088030, -0.033048, -0.323671, -0.206566, 0.109279, -0.046077, -0.020657, -0.241265, 0.019414, -0.073760, 0.066414, -0.211693, 0.038956, 0.063885, -0.195446, 0.066106, 0.050598, -0.043393, -0.088878, -0.045683, -0.038408, 0.270234, 0.008195, -0.208056, -0.021953, -0.007816, -0.040012, 0.112660, -0.039505, -0.060729, 0.108189, -0.066276, 0.034591, -0.025648, 0.038717, 0.072765, -0.218895, -0.062543, 0.008313, -0.013547, 0.038047, -0.079340, -0.115802, -0.108748, 0.160478, -0.070355, -0.013505, 0.001732, -0.130134, -0.066000, 0.001894, 0.061890, -0.106928, 0.035437, -0.105427, -0.036172, 0.077568, -0.028998, 0.109493, 0.198249, -0.152100, -0.247378, -0.007727, -0.040935, 0.018899, 0.020369, -0.116417, -0.037932, 0.040718, 0.019449, 0.088794, -0.014703, 0.060364, 0.054907, 0.002984, 0.039335, 0.064052, 0.176181, -0.131771, -0.071442, -0.075966, -0.091585, -0.084284, 0.048288, -0.057182, -0.134468, 0.131080, -0.220348, 0.010528, -0.024509, -0.000072, -0.121092, -0.061675, 0.102668, 0.088580, 0.074090, -0.073343, 0.124642, 0.037529, -0.150481, -0.060720, 0.045388, 0.045435, 0.110100, 0.111971, 0.062229, 0.038525, 0.139536, -0.126165, 0.054674, 0.217674, -0.118082, -0.205816, -0.041321, 0.132137, -0.038668, 0.087049, 0.133630, -0.126031, 0.062563, -0.103972, -0.011177, -0.038011, -0.080902, -0.059126, -0.054810, -0.183830, 0.095315, -0.126425, -0.040395, -0.183642, -0.290865, 0.208608, 0.085437, 0.133955, -0.024208, 0.057405, -0.045685, -0.104377, 0.050633, -0.030613, -0.030200, -0.013165, 0.029857, -0.006969, 0.122675, 0.053355, 0.078943, 0.170076, 0.083059, 0.017627, -0.085854, -0.083754, 0.082734, 0.205185, -0.045534, -0.106840, -0.024386, -0.235401, -0.009400, -0.143873, 0.133664, -0.079488, -0.006170, -0.043987, 0.076200, -0.006393, 0.006480, -0.060405, -0.052390, 0.132738, -0.048855, 0.126041, -0.013201, 0.109502, 0.105537, 0.208539, -0.082225, 0.111868, 0.021789, -0.063445, -0.119760, -0.018942, -0.019072, 0.025736, 0.140829, -0.115631, -0.037773, 0.014081, -0.084708, -0.059725, 0.081242, -0.218649, 0.059062, 0.036030, -0.044041, -0.029177, 0.061121, 0.251979, 0.045340, 0.037397, -0.058674, 0.075205, 0.032006, -0.023324, 0.002139, -0.263666, -0.086502, -0.000375, 0.062516, -0.188990, 0.034800, -0.062739, 0.030045, -0.085034, 0.066524, -0.094228, 0.072402, -0.040011, -0.116920, 0.132721, 0.108039, 0.038653, -0.095025, 0.031889, 0.019984, -0.053344, -0.053428, -0.002135, -0.127179, -0.092878, -0.064620, 0.095156, 0.178577, -0.042061, -0.010872, -0.076603, 0.050656, -0.260375, 0.066181, -0.048176, 0.009354, -0.169570, -0.057745, 0.060972, -0.035799, -0.061200, 0.070321, -0.035491, 0.186448, -0.131719, -0.112950, -0.074425, 0.075744, -0.181505, -0.103242, 0.214975, -0.055409, 0.012402, -0.104753, -0.151273, 0.127548, -0.020574, -0.006788, -0.057447, -0.146018, 0.049620, 0.105499, -0.033810, 0.128839, 0.047308, 0.106148, 0.096299, 0.092840, 0.049419, 0.284900, -0.065311, -0.129898, -0.005343, -0.018969, -0.114778, 0.135392, -0.058236, -0.250104, 0.218009, 0.172885, -0.108120, -0.016687, -0.038899, 0.027521, -0.038984, 0.045571, 0.173847, -0.152789, 0.179525, -0.045752, 0.102328, -0.329427, 0.098693, 0.172111, -0.201866, -0.016191, -0.191308, -0.046811, -0.056715, -0.041992, -0.121772, 0.126827, 0.085615, 0.014094, 0.132441, -0.021916, 0.058572, -0.015446, -0.094244, -0.026883, -0.102480, -0.027156, 0.121651, 0.089729, 0.139715, -0.348230, 0.003583, -0.016937, 0.260762, 0.095857, -0.099383, 0.025118, -0.127208, -0.000859, 0.089542, -0.063848, 0.003463, 0.249191, 0.289359, 0.069229, 0.036740, 0.043097, -0.103791, -0.004500, -0.127035, 0.019432, -0.047438, 0.226935, 0.098894, 0.071871, 0.148865, -0.024241, -0.021783, -0.047939, 0.063671, 0.133419, 0.018413, -0.021382, -0.044696, -0.006940, 0.677183, -0.142716, -0.110133, -0.202088, -0.036368, 0.077196, 0.036584, -0.019362, -0.008300, -0.148631, 0.143913, -0.088931, 0.047510, 0.069454, 0.163853, 0.081567, 0.069268, -0.096887, 0.077159, -0.004841, 0.077025, 0.055366, 0.053573, -0.137808, 0.014546, 0.044704, -0.034603, 0.081974, 0.145496, -0.194725, -0.037113, -0.018032, -0.083485, -0.050184, 0.098446, -0.127761, -0.009436, -0.152586, 0.009146, -0.024900, 0.059450, -0.068477, 0.163836, 0.067859, -0.068912, -0.131303, -0.032480, -0.070498, 0.058060, -0.040360, 0.129701, -0.151337, 0.114591, -0.021017, -0.029139, 0.050756, 0.256466, -0.020757, -0.023750, -0.097589, 0.006947, -0.022741, 0.103667, 0.048112, -0.139119, -0.023686, 0.042073, 0.142537, 0.037976, -0.026353, -0.085442, -0.028076, 0.085605, -0.073952, 0.164115, 0.099156, -0.034135, -0.092651, -0.244873, -0.209171, 0.096888, 0.128501, 0.284937, -0.119141, 0.110726, 0.038952, -0.174968, 0.135266, 0.048078, 0.057075, 0.062593, -0.018037, -0.008031, -0.044985, -0.106926, -0.007957, 0.116766, 0.008355, 0.002351, 0.117693, 0.323468, 0.035274, 0.058384, -0.006341, 0.026755, 0.037961, 0.002410, 0.026927, -0.134846, -0.056251, -0.044381, 0.084509, -0.049086, -0.021632, -0.009278, -0.119065, 0.092700, -0.227267, 0.002568, 0.068348, 0.148723, 0.047690, 0.001633, -0.060685, 0.021056, 0.088190, -0.040228, -0.141415, -0.134729, -0.014143, -0.113032, -0.389992, 0.017368, 0.000233, 0.036049, 0.100001, -0.039815, 0.031219, -0.009520, 0.020167, -17.560251, 0.031402, 0.062646, -0.000305, 0.000351, -0.121338, 0.041774, -0.066650, 0.019267, 0.120415, -0.132900, 0.105727, -0.126407, -0.059157, -0.091393, -0.023915, -0.111860, 0.052003, -0.084105, -0.095778, -0.019815, 0.295927, -0.146690, 0.184004, -0.121180, -0.122926, 0.187363, 0.141952, -0.242966, -0.149112, -0.142458, 0.021671, 0.104152, 0.043685, 0.256449, 0.039963, 0.019566, 0.146123, -0.101624, 0.147438, 0.149866, 0.260162, -0.048283, -0.129161, -0.057012, 0.019565, 0.055049, 0.084132, -0.086373, -0.047386, -0.051909, -0.084550, 0.026338, -0.177579, 0.057977, 0.004294, 0.032369, -0.063187, -0.004218, 0.203801, 0.104988, 0.044866, -0.115330, 0.001208, -0.075143, -0.018574, 0.003123, 0.144965, 0.076498, -0.107642, -0.087044, -0.285483, -0.068918, -0.038863, 0.129634, -0.509720, -0.057580, 0.074435, -0.329218, -0.091229, -0.012547, -0.250338, -0.048831, 0.009125, -0.013701, -0.135358, -0.103943, 0.122000, 0.015015, 0.186748, 0.013945, 0.107904, 0.072840, -0.093424, -0.034193, -0.368991, 0.099667, 0.107972, -0.097647, 0.020200, 0.202852, 0.003143, -0.023519, -0.192420, 0.105191, 0.019366, 0.085177, 0.222103, 0.009125, 0.269473, -0.240621, -0.045546, -0.144950, 0.102957, -0.042734, 0.219884, 0.006519, -0.101173, -0.010339, -0.154858, -0.144933, -0.087029, -0.056886, 0.112035, 0.100917, -0.080368, -0.015355, 0.084853, 0.221549, 0.001151, -0.146819, -0.007874, 0.052456, 0.015544, -0.018728, 0.125240, 0.053811, 0.043976, -0.019695, 0.017176, 0.225304, 0.032954, -0.223059, 0.127289, 0.207704, 0.125859, 0.075234, 0.083075, -0.062068, -0.094499, -0.044568, 0.114748, 0.017832, -0.138829, -0.045451, 0.031061, -0.110023, -0.025246, -0.020514, 0.064687, 0.133305, -0.046727, 0.126711, 0.000801, 0.008386, -0.069578, -0.033664, -0.118024, -0.000837, 0.140300, 0.115271, -0.160692, 0.174190, -0.045878, 0.018823, 0.163019, -0.094100, -0.023819, -0.082772, -0.100585, 0.033611, 0.055864, 0.119997, -0.151538, 0.127290, 0.117162, -0.031406, -0.101627, 0.154954, 0.207332, -0.171334, 0.220817, -0.107421, 0.062537, -0.059915, -0.039356, -0.140241, -0.049766, -0.124542, -0.229954, 0.010013, -0.123433, -0.113180, 0.203433, -0.046592, -0.051765, 0.028296, 0.130263, 0.032389, 0.193054, -0.034520, -0.207641, 0.030356, -0.024933, 0.046441, -0.109342, 0.010195, -0.062383, -0.127414, 0.083187, -0.033987, -0.094084, -0.022853, -0.044369, -0.101048, -0.070643, -0.347899, -0.006629, -0.143909, -0.096722, 0.006665, -0.069368, -0.049028, -0.056009, 0.180570, 0.109365, -0.102656, 0.062361, -0.270445, 0.047035, 0.020955, 0.050168, -0.198015, 0.101712, -0.130133, -0.057636, 0.049657, -0.065991, -0.054406, -0.021417, 0.114409, 0.019887, 0.038968, 0.083234, 0.014027, 0.072576, -0.190975, 0.058188, -0.065989, 0.078768, -0.166600, -0.017930, -0.151702, 0.107779, -0.045874, 0.193601, -0.036514, 0.016187, -0.019196, 0.075741, 0.074752, -0.014977, -0.127832, 0.110609, 0.165647, 0.137579, -0.146175, 0.158391, -0.053769, -0.060472, 0.043749, -0.079397, -0.289111, 0.096437, 0.034058, -0.067546, -0.132831, 0.034298, 0.091708, 0.267784, 0.008844, -0.031677, 0.061472, -0.217702, -0.051128, 0.105082, 0.087476, 0.111178, 0.042027, 0.089634, -0.321581, 0.030524, -0.052669, -0.057072, 0.209049, 0.072233, 0.042392, 0.016775, -0.081067, -0.092204, 0.014861, 0.043793, -0.059614, 0.062412, -0.031823, -0.108595, -0.024099, 0.115535, 0.058593, -0.206247, -0.071850, 0.007350, -0.184772, -0.028674, 0.134175, -0.090862, -0.116007, 0.161286, 0.146457, -0.129311, 0.264008, -0.015322, 0.063028, 0.125847, 0.186666, 0.072334, -0.155329, -0.049842, 0.131537, -0.346177, 0.020658, -0.053742, -0.048936, -0.030947, 0.071958, 0.225631, -0.047669, 0.281787, -0.318682, 0.051802, -0.560169, -0.041564, 0.006527, -0.243088, -0.121814, 0.022289, -0.064393, 0.222421, 0.106093, 0.144770, -0.021951, 0.053686, 0.039989, 0.012293, -0.117027, 0.109881, 0.037751, -0.006731, 0.052598, 0.108269, -0.199924, 0.109416, -0.106555, -0.116143, -0.124453, -0.061717, -0.147258, 0.050344, 0.024112, 0.215614, 0.042398, -0.043336, -0.037669, 0.142949, -0.003854, -0.205387, 0.039217, -0.155085, 0.182128, 0.100197, -0.015709, 0.129793, 0.080563, 0.046750, 0.357068, 0.142408, 0.029890, 0.006892, -0.232900, -0.130439, 0.085445, -0.124606, -0.004155, 0.071344, -0.087067, 0.010737, 0.071014, -0.113435, 0.158935, -0.155636, 0.157870, 0.159434, 0.132377, -0.140437, -0.140162, 0.010436, 0.157148, 0.212509, -0.096184, 0.037706, 0.142679, 0.087406, -0.226862, 0.229564, -0.001350, -0.059905, 0.061493, -0.119524, -0.136959, -0.106491, 0.114841, 0.183394, 0.058116, -0.063762, -0.083453, -0.048704, 0.002241, -0.018818, 0.214347, 0.104077, -0.079427, -0.083980, -0.136077, -0.028594, -0.025977, 0.015855, -0.152728, -0.047104, 0.003986, -0.039404, 0.024761, 0.135967, -0.005845, 0.060347, 0.068921, 0.080455, -0.047108, -0.087921, 0.105101, -0.002233, -0.054010, 0.089830, -0.001870, -0.029980, -0.129639, -0.077746, 0.134815, 0.021763, -0.064423, -0.032674, -0.114967, 0.038400, -0.139010, 0.101270, 0.086643, -0.255134, -0.030313, 0.059016, 0.065511, 0.191318, 0.119068, -0.142207, -0.007195, 0.153521, -0.102213, 0.068673, 0.040845, 0.119193, -0.023874, -0.101078, -0.118411, -0.168753, -0.012316, 0.343408, -0.022840, 0.035783, 0.008199, -0.062494, 0.032484, 0.006892, 0.011465, -0.009145, 0.099642, 0.003002, -0.006301, 0.121277, 0.085010, -0.097261, 0.011950, -0.092597, 0.128223, 0.010209, -0.051851, 0.006322, -0.148499, -0.085036, 0.186526, -0.067553, 0.035917, 0.068583, 0.092601, 0.062599, -0.090954, -0.006533, -0.019775, 0.029196, -0.103416, -0.029282, -0.133812, -0.117944, -0.032113, -0.005479, -0.028942, 0.058452, -0.125991, 0.048418, 0.011948, -0.050808, -0.142183, -0.236945, 0.158152, -0.049617, 0.192856, 0.047404, 0.090607, -0.165725, 0.177486, -0.014033, 0.131047, 0.031389, -0.133362, -0.188347, -0.234466, -0.026319, -0.042951, 0.138424, -0.013183, 0.037700, 0.035605, -0.035174, 0.055972, -0.307531, 0.232464, -0.059816, -0.088053, -0.190345, -0.066762, -0.017807, -0.063804, -0.028137, 0.057324, -0.059485, -0.024240, -0.006635, -0.051617, 0.114749, -0.043398, -0.015006, 0.018601, -0.184944, -0.196960, 0.228250, -0.083176, -0.151567, -0.171002, 0.170817, -0.093068, 0.091797, 0.043906, 0.003094, -0.077643, 0.021524, 0.034549, 0.126353, -0.073374, 0.023899, 0.028393, 0.191842, -0.060294, 0.103405, -0.036983, -0.052341, 0.037159, -0.299527, -0.071038, 0.090827, -0.038913, 0.026257, -0.187989, 0.003810, -0.113830, -0.050896, 0.019581, 0.041997, 0.003261, -0.055369, 0.156824, -0.058390, -0.075181, 0.062765, 0.120935, -0.130294, 0.094855, 0.259560, -0.090834, 0.052287, -0.113942, 0.088777, 0.019882, -0.107042, 0.115183, -0.189182, 0.006723, -0.131681, 0.008823, 0.002470, -0.134018, -0.065393, -0.041999, 0.042253, 0.170972, 0.053821, -0.013449, -0.083807, 0.081735, -0.193318, 0.053062, 0.127185, -0.192140, -0.000839, -0.061118, -0.112899, 0.269778, 0.005209, 0.033114, -0.084155, 0.058585, -0.001494, -0.039173, -0.158692, 0.051888, -0.040920, -0.088873, 0.138005, -0.094690, -0.136133, 0.012102, 0.023572, -0.169486, -0.055359, 0.061275, 0.129436, 0.059882, 0.045207, 0.164967, 0.177469, 0.481251, 0.054807, 0.071716, 0.030825, 0.132782, 0.061844, 0.130011, 0.102832, 0.014541, 0.219767, -0.080910, 0.198117, -0.216629, -0.130676, -0.014194, -0.106681, 0.003732, 0.125827, -0.162662, -0.199298, -0.094990, 0.046070, 0.065603, -0.063598, 0.054432, -0.147039, -0.116275, 0.038240, -0.048517, -0.147684, -0.185406, -0.036768, 0.153569, 0.007226, -0.059407, 0.005382, 0.157409, -0.025293, 0.065589, -0.061700, 0.220015, 0.051336, -0.125858, -0.111602, 0.099379, -0.012292, 0.108578, 0.056446, 0.133125, 0.126082, -0.055100, -0.069437, -0.047941, 0.012160, -0.107565, -0.022220, 0.073833, 0.227920, -0.050044, -0.151612, -0.086171, -0.092477, 0.123581, -0.099818, -0.017800, -0.044686, -0.126607, -0.139814, 0.108080, -0.007431, -0.056303, 0.001033, 0.142660, 0.074813, 0.058472, -0.084213, -0.153521, 0.037884, -0.021679, 0.016159, -0.009723, -0.011298, 0.118332, -0.130813, -0.029906, -0.110681, 0.066274, -0.067938, 0.123664, 0.169787, -0.175018, 0.057734, -0.042671, 0.131462, -0.010378, 0.250109, -0.078314, 0.106315, 0.045705, -0.059676, 0.111006, -0.030596, -0.062873, 0.071176, 0.045725, 0.060742, -0.094350, -0.116889, -0.005298, 0.109534, 0.020845, 0.035422, 0.087789, 0.022662, -0.139119, -0.034978, 0.140012, -0.000138, 0.062438, 0.060544, -0.143456, 0.159155, -0.080226, 0.064854, 0.017204, 0.078555, -0.312732, -0.049333, -0.114972, -0.054956, -0.046951, -0.069219, 0.088113, -0.125752, -0.098918, -0.016828, 0.061117, -0.023489, 0.262428, -0.113538, 0.080467, 0.004451, 0.064655, -0.166613, 0.068366, 0.223530, 0.038508, 0.134115, 0.199807, 0.227365, 0.038864, 0.142108, 0.067712, 0.015172, 0.069827, 0.081276, -0.033247, -0.144224, 0.110971, -0.125569, -0.028171, 0.022167, -0.023597, -0.059744, -0.093713, -0.003360, -0.090202, 0.113533, -0.040930, -0.151915, 0.183841, -0.029404, -0.081949, 0.083049, -0.075860, 0.076466, -0.027114, -0.207886, 0.163120, 0.151910, -0.085722, 0.035349, -0.159772, -0.109084, 0.195695, 0.005579, 0.158401, -0.037048, -0.043944, 0.061840, -0.075777, 0.102455, -0.136061, 0.015539, -0.024582, -0.050807, -0.028741, 0.067760, -0.051109, 0.019125, 0.053387, -0.186889, 0.144725, -0.061923, -0.020042, 0.056338, 0.041101, 0.053671, -0.048678, 0.067230, -0.135270, -0.004464, -0.040022, 0.127166, 0.089554, 0.025853, -0.119083, 0.037060, 0.092043, 0.050736, 0.145814, 0.210574, 1.307182, -0.080285, -0.133174, 0.044181, -0.006088, 0.171834, -0.067296, 0.227449, 0.153793, -0.062197, 0.100289, 0.112550, 0.029630, -0.238683, 0.112148, -0.024818, 0.197540, 0.049651, -0.103266, -0.038647, -0.023400, -0.095559, 0.085548, -0.049817, -0.016849, 0.214791, -0.020292, 0.054894, -0.174885, -0.063816, 0.005847, 0.107943, -0.143944, 0.127508, -0.061392, -0.124929, 0.124236, 0.000747, -0.080838, -0.034588, 0.019234, -0.103349, 0.055600, -0.103468, 0.072132, -0.114047, -0.051017, -0.147391, 0.010201, -0.118329, 0.245386, -0.025556, -0.203400, -0.070726, 0.106067, 0.102245, 0.071107, -0.039556, -0.004303, 0.151525, 0.088495, -0.052666, 0.099617, 0.166001, 0.132955, 0.004214, 0.138000, 0.021966, 0.117119, -0.137730, -0.001686, -0.152241, 0.103399, -0.121396, 0.237599, -0.017448, -0.049311, 0.048781, -0.031326, 0.135011, -0.023239, -0.023123, -0.064248, 0.145788, 0.094706, -0.066675, -0.103344, 0.013480, 0.088714, -0.057631, 0.102930, -0.186519, 0.067461, -0.065874, -0.006699, -0.008791, 0.022082, 0.021527, -0.054976, 0.050465, 0.067240, 0.107685, 0.048799, 0.008721, 0.190600, 0.042695, 0.035957, 0.075790, 0.024237, 0.042844, 0.017406, 0.050326, -0.206489, -0.174855, 0.028568, 0.074619, 0.044658, -0.106416, -0.009997, -0.059396, -0.062128, 0.002757, -0.023875, 0.093777, 0.016492, -0.422595, -0.025214, 0.072371, 0.126453, -0.003051, 0.107823, 0.050557, -0.039889, -0.292374, 0.080554, 0.009501, -0.030588, -0.004588, -0.158899, 0.133628, -0.159994, 0.048548, 0.001413, 0.078073, 0.098567, -0.010438, 0.012715, -0.016493, -0.139262, 0.061775, -0.167695, 0.291744, 0.069400, 0.016622, 0.015490, 0.003041, -0.042024, 0.055067, 0.095599, 0.030203, 0.078127, 0.093718, -0.136859, -0.119588, -0.186097, -0.014693, -0.069448, -0.030063, -0.035874, 0.218185, -0.121556, 0.056818, -0.019789, -0.131317, 0.031296, -0.107446, -0.065685, -0.067036, 0.052630, -0.005172, -0.107737, -0.160952, 0.038470, -0.044669, 0.019512, 0.065464, 0.075660, -0.116097, 0.103803, -0.070672, 0.036499, 0.006734, 0.021789, 0.004654, 0.061178, 0.151650, 0.043863, -0.176501, 0.076531, -0.121777, -0.021610, 0.083082, 0.175232, -0.051474, -0.132132, -0.100258, 0.049386, 0.124653, -0.372808, -0.023726, -0.013614, 0.021540, 0.053970, -0.031760, 0.058089, -0.080102, 0.026285, -0.023720, -0.151576, 0.062510, 0.046695, -0.108900, -0.150990, 0.058909, 0.040689, 0.062452, 0.032578, -0.009215, 0.063232, -0.012985, -0.082494, 0.080091, -0.223397, -0.267580, -0.071734, -0.187211, -0.175497, -0.063138, 0.089356, 0.088037, -0.150269, 0.108458, 0.111737, -0.043819, 0.066353, -0.091624, -0.003232, -0.084642, -0.030678, 0.108811, -0.139834, 0.060631, -0.078275, 0.187151, 0.107771, 0.102642, -0.012629, 0.003733, 0.073110, 0.024390, 0.034245, -0.212272, 0.158737, -0.087567, -0.209491, -0.023604, -0.019122, -0.092557, 0.033499, -0.193531, -0.049179, -0.131268, 0.001090, 0.104372, -0.085945, 0.103111, -0.008147, 0.032869, 0.053545, 0.046363, 0.004164, -0.025531, 0.086678, 0.089752, 0.035347, -0.036365, 0.032151, 0.082001, 0.085604, 0.063768, -0.197579, 0.007230, 0.123480, 0.068322, 0.229867, -0.019057, -0.049470, -0.014650, -0.142659, 0.108674, -0.054853, -0.059361, 0.034454, -0.045122, -0.035088, 0.043680, -0.266513, -0.068722, 0.043407, -0.030239, 0.013062, -0.121207, 0.034871, 0.137072, -0.060138, -0.066898, -0.037768, -0.033868, 0.026119, 0.004924, -0.103202, -0.094462, 0.040436, 0.120147, 0.125521, -0.066460, -0.047309, 0.014630, 0.129495, 0.109331, 0.073543, -0.058197, 0.060052, -0.030361, -0.091179, 0.002897, 0.006175, 0.014321, 0.023485, -0.176344, -0.006739, 0.027919, 0.018064, -0.276239, -0.024107, 0.106251, 0.010355, -0.033997, -0.100764, -0.110482, 0.002556, -0.010032, 0.035186, 0.121912, -0.065353, 0.094416, 0.004629, 0.029290, 0.132925, 0.006140, 0.008870, 0.065623, 0.048869, 0.093244, 0.080586, -0.164347, -0.097979, -0.231322, 0.121290, -0.002545, 0.006872, 0.011683, 0.125875, -0.084730, -0.133168, 0.038148, -0.219887, 0.093054, -0.119322, -0.098027, 0.103942, 0.034731, -0.056183, -0.104774, -0.121926, -0.178987, 0.003817, -0.128040, -0.049184, -0.037602, 0.148632, -0.047410, -0.124279, -0.011996, -0.159390, 0.135262, 0.052034, -0.048486, -0.007119, 0.036029, -0.118560, -0.056069, -0.162273, -0.133867, 0.020461, 0.145542, 0.170790, 0.063234, 0.234156, 0.009867, -0.029983, 0.017902, 0.136641, 0.029571, -0.075318, -0.085281, 0.106845, 0.085930, -0.036203, -0.050606, -0.075667, 0.099817, 0.063058, 0.002306, 0.121536, -0.026002, -0.145052, 0.018436, -0.208630, -0.013625, 0.277756, -0.077494, 0.032889, 0.008865, -0.026560, 0.018103, -0.140435, -0.124794, 0.114586, -0.113241, 0.152404, 0.122089, 0.080138, -0.176922, -0.022841, 0.008654, 0.067025, -0.183390, 0.146847, -0.042429, 0.103191, -0.144039, -0.049146, 0.031298, 0.035034, 0.084029, 0.095841, -0.013510, -0.042343, 0.157538, -0.022027, -0.050378, -0.025126, -0.130541, -0.000940, 0.118086, 0.109244, -0.067703, -0.119353, -0.006909, 0.075037, -0.111312, -0.031690, 0.047065, -0.144276, 0.124092, 0.188607, -0.094470, -0.191534, 0.062763, 0.154428, 0.021396, 0.011102, -0.067822, 0.021463, -0.045017, 0.092374, -0.097997, -0.197976, 0.124790, -0.149524, -0.109539, -0.009514, -0.001633, -0.025193, -0.076272, -0.142697, -0.048958, 0.088766, 0.047299, 0.102958, 0.043278, 0.015998, 0.265931, 0.216833, -0.010011, 0.016229, -0.099601, -0.131450, -0.150390, 0.126453, -0.111937, -0.052300, 0.177407, 0.115467, -0.113335, 0.068112, 0.038659, 0.058132, 0.079879, -0.003602, -0.022686, -0.161627, -0.084063, -0.008425, -0.044294, -0.161736, -0.037143, 0.115853, 0.072605, 0.151796, -0.166894, 0.043029, 0.040400, 0.102378, -0.055698, -0.006078, -0.101925, -0.069470, 0.023171, -0.378824, 0.040074, -0.011152, -0.008271, 0.182452, -0.156884, 0.087277, -0.036570, -0.010507, 0.003790, -0.043893, 0.047902, 0.148466, -0.048559, 0.001693, -0.117694, -0.128362, -0.046627, 0.031107, 0.085624, -0.045252, -0.054078, 0.041979, 0.114015, -0.003651, 0.045461, -0.015073, 0.006541, -0.325775, 0.053410, 0.084582, 0.221443, -0.102977, -0.084458, 0.015049, -0.058087, -0.007276, -0.063492, 0.014074, 0.255791, -0.058815, -0.129200, -0.146628, -0.084106, 0.048064, 0.022999, -0.203971, -0.049912, -0.005668, -0.187804, -0.010728, -0.125252, 0.169264, -0.055967, 0.161077, 0.119071, 0.033876, -0.021196, -0.031746, -0.105268, 0.058991, -0.117101, -0.042655, 0.119778, -0.128045, 0.042478, 0.107519, -0.182803, 0.039299, -0.087434, -0.060082, 0.019341, 0.029234, -0.121861, 0.017733, 0.074250, -0.173260, 0.164209, -0.159713, 0.117683, 0.052416, 0.202190, 0.046435, -0.092751, 0.030402, 0.076513, -0.138443, 0.097043, -0.036856, -0.008675, 0.035766, 0.218701, -0.055522, -0.046284, -0.214577, 0.054488, -0.012900, -0.174178, -0.125368, -0.144828, 0.196536, 0.032619, -0.259316, 0.013355, -0.074827, -0.143640, -0.006778, -0.069064, -0.027958, -0.030783, 0.008034, -0.030716, 0.035041, 0.039476, 0.005347, -0.015996, -0.039156, -0.064262, 0.086718, 0.047876, 0.025550, -0.212628, 0.007176, 0.033534, -0.041059};
//     std::vector<int32_t> src1data = {1};
//     std::vector<float> dstdata(4096, 0.0);
// 
//     void* selfDeviceAddr = nullptr;
//     void* otherDeviceAddr = nullptr;
//     void* outDeviceAddr = nullptr;
//     std::vector<int64_t> src0Shape = {1, 1, 2, 4096};
//     std::vector<int64_t> src1Shape = {1, 1, 1, 1};
//     std::vector<int64_t> dstShape = {1, 1, 1, 4096};
//     ret = data_addr_malloc(src0Shape, src0data, &selfDeviceAddr);
//  
//     ret = data_addr_malloc(src1Shape, src1data, &otherDeviceAddr);
// 
//     ret = data_addr_malloc(dstShape, dstdata, &outDeviceAddr);
// 
// 
//     src0->data = selfDeviceAddr;
//     src1->data = otherDeviceAddr;
//     dst->data = outDeviceAddr;
//     src0->type = GGML_TYPE_F32;
//     src1->type = GGML_TYPE_I32;
//     dst->type = GGML_TYPE_F32;
//     ggml_ascend_get_rows(*ctx, dst);
//     std::vector<float> resultData(4096, 0);
//     ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
//                         4096 * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
//     FILE * f = fopen("out.txt", "w");
//     fprintf(f, "{");
//     for (int i = 0; i < 4096; i++) {
//         fprintf(f, "%f, ", resultData[i]);
//     }
//     fprintf(f, "}\n");
//     fclose(f);
// }