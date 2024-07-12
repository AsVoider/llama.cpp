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
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int create_acl_tensor(const aclnn_shape_t& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor) {
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}