//
// Created by 35763 on 2024/6/26.
//

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "common.h"

#include <cstdint>

aclDataType ggml_to_acl_map[GGML_TYPE_COUNT] = {
    ACL_FLOAT,        // GGML_TYPE_F32
    ACL_FLOAT16,      // GGML_TYPE_F16
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_0
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_1
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_2 (removed)
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_3 (removed)
    ACL_DT_UNDEFINED, // GGML_TYPE_Q5_0
    ACL_DT_UNDEFINED, // GGML_TYPE_Q5_1
    ACL_DT_UNDEFINED, // GGML_TYPE_Q8_0
    ACL_DT_UNDEFINED, // GGML_TYPE_Q8_1
    ACL_DT_UNDEFINED, // GGML_TYPE_Q2_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q3_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q4_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q5_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q6_K
    ACL_DT_UNDEFINED, // GGML_TYPE_Q8_K
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ2_XXS
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ2_XS
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ3_XXS
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ1_S
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ4_NL
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ3_S
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ2_S
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ4_XS
    ACL_INT8,         // GGML_TYPE_I8
    ACL_INT16,        // GGML_TYPE_I16
    ACL_INT32,        // GGML_TYPE_I32
    ACL_INT64,        // GGML_TYPE_I64
    ACL_DOUBLE,       // GGML_TYPE_F64
    ACL_DT_UNDEFINED, // GGML_TYPE_IQ1_M
    ACL_BF16          // GGML_TYPE_BF16
};

size_t ggml_type_size_t[GGML_TYPE_COUNT] = {
    sizeof(float),      // GGML_TYPE_F32
    sizeof(short),      // GGML_TYPE_F16
    1,                  // GGML_TYPE_Q4_0
    1,                  // GGML_TYPE_Q4_1
    1,                  // GGML_TYPE_Q5_0
    1,                  // GGML_TYPE_Q5_1
    1,                  // GGML_TYPE_Q8_0
    1,                  // GGML_TYPE_Q8_1
    1,                  // GGML_TYPE_Q2_K
    1,                  // GGML_TYPE_Q3_K
    1,                  // GGML_TYPE_Q4_K
    1,                  // GGML_TYPE_Q5_K
    1,                  // GGML_TYPE_Q6_K
    1,                  // GGML_TYPE_Q8_K
    1,                  // GGML_TYPE_IQ2_XXS
    1,                  // GGML_TYPE_IQ2_XS
    1,                  // GGML_TYPE_IQ3_XXS
    1,                  // GGML_TYPE_IQ1_S
    1,                  // GGML_TYPE_IQ4_NL
    1,                  // GGML_TYPE_IQ3_S
    1,                  // GGML_TYPE_IQ2_S
    1,                  // GGML_TYPE_IQ4_XS
    sizeof(char),       // GGML_TYPE_I8
    sizeof(short),      // GGML_TYPE_I16
    sizeof(int),        // GGML_TYPE_I32
    sizeof(long long),  // GGML_TYPE_I64
    sizeof(double),     // GGML_TYPE_F64
    1,                  // GGML_TYPE_IQ1_M
    2                   // GGML_TYPE_BF16
};


int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  // ret = aclrtCreateContext(context, deviceId);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  // ret = aclrtSetCurrentContext(*context);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  // ret = aclrtCreateStream(stream);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int create_acl_tensor(const aclnn_shape_t& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor, size_t *nb) {
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if(nb) {
        for (int64_t i = 0; i < shape.size(); i++) {
          strides[shape.size() - 1 - i] = nb[i] / aclDataTypeSize(dataType);
          // LOG_PRINT("strides[%ld] is: %ld\n", i, strides[i]);
        }
    } else {
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
          // LOG_PRINT("strides[%ld] is: %ld\n", i, strides[i]);
      }
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int addr_malloc(const aclnn_shape_t& shape, void** deviceAddr, size_t size_t) {
    auto size = GetShapeSize(shape) * size_t;
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    return 0;
}