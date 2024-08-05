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

  GGML_UNUSED(context);
  // ret = aclrtCreateContext(context, deviceId);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  // ret = aclrtSetCurrentContext(*context);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  // ret = aclrtCreateStream(stream);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  ((void *)stream);
  return 0;
}

int create_acl_tensor(const aclnn_shape_t& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor, size_t *nb) {
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if(nb) {
        for (decltype(shape.size()) i = 0; i < shape.size(); i++) {
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

aclTensor * ggml_ascend_create_tensor(const ggml_tensor * tensor, int64_t * ne, size_t* nb, int64_t dims,
                             aclFormat format,
                             size_t offset) {
  int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

  int64_t acl_storage_len(0);
  if (ne == nullptr) {
    acl_storage_len = ggml_nbytes(tensor);
    for (auto i(0); i < GGML_MAX_DIMS; i++) {
      acl_ne[i] = tensor->ne[i];
      acl_stride[i] = tensor->nb[i] / ggml_element_size(tensor);
    }
  } else {
    for (auto i(0); i < dims; i++) {
      acl_storage_len += (ne[i] - 1) * nb[i];
      acl_ne[i] = ne[i];
      acl_stride[i] = nb[i] / ggml_element_size(tensor);
    }
  }


  int64_t final_dims(dims == 0 ? GGML_MAX_DIMS : dims);
  std::reverse(acl_ne, acl_ne + final_dims);
  std::reverse(acl_stride, acl_stride + final_dims);

  auto acl_tensor = aclCreateTensor(
    acl_ne, final_dims, ggml_to_acl_map[tensor->type], acl_stride,
    offset / ggml_element_size(tensor), format, &acl_storage_len, 1,
    tensor->data
  );

  return acl_tensor;
}

aclTensor* ggml_ascend_create_tensor(void* data_ptr, aclDataType dtype,
                                   size_t type_size, int64_t* ne, size_t* nb,
                                   int64_t dims, aclFormat format,
                                   size_t offset) {
    int64_t tmp_ne[GGML_MAX_DIMS * 2];
    int64_t tmp_stride[GGML_MAX_DIMS * 2];

    memcpy(tmp_ne, ne, dims * sizeof(int64_t));
    for (int i = 0; i < dims; i++) {
        tmp_stride[i] = nb[i] / type_size;
    }

    std::reverse(tmp_ne, tmp_ne + dims);
    std::reverse(tmp_stride, tmp_stride + dims);

    int64_t acl_storage_len = 0;
    for (int i = 0; i < dims; i++) {
        acl_storage_len += (ne[i] - 1) * nb[i];
    }

    aclTensor* acl_tensor =
        aclCreateTensor(tmp_ne, dims, dtype, tmp_stride, offset / type_size,
                        format, &acl_storage_len, 1, data_ptr);

    return acl_tensor;
}

int64_t ggml_cann_get_bcast_shape(const ggml_tensor* src0,
                                  const ggml_tensor* src1,
                                  int64_t* bcast_src0_ne,
                                  int64_t* bcast_src1_ne, size_t* bcast_src0_nb,
                                  size_t* bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    int bcast_dim_cnt = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        if (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] *
                                           bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] *
                                           bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(
    const int64_t* input_ne, const int64_t* weight_ne, const int64_t* dst_ne,
    const size_t* input_nb, const size_t* weight_nb, const size_t* dst_nb,
    int64_t* bcast_input_ne, int64_t* bcast_weight_ne, int64_t* bcast_dst_ne,
    size_t* bcast_input_nb, size_t* bcast_weight_nb, size_t* bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(input_ne[2] == dst_ne[2]);
    GGML_ASSERT(input_ne[3] == dst_ne[3]);

    int bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = input_ne[i] / weight_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        if (i < 2 || nr == 1) {
            bcast_input_ne[bcast_dim_cnt] = input_ne[i];
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_dst_ne[bcast_dim_cnt] = dst_ne[i];

            bcast_input_nb[bcast_dim_cnt] = input_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dst_nb[bcast_dim_cnt] = dst_nb[i];
            bcast_dim_cnt++;
        } else {
            // Need to add an extra dim.
            bcast_input_ne[bcast_dim_cnt] = nr;
            bcast_dst_ne[bcast_dim_cnt] = nr;
            bcast_weight_ne[bcast_dim_cnt] = 1;
            bcast_input_nb[bcast_dim_cnt] = input_nb[i];
            bcast_dst_nb[bcast_dim_cnt] = dst_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dim_cnt++;

            bcast_input_ne[bcast_dim_cnt] = input_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt] = dst_ne[i] / nr;
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_input_nb[bcast_dim_cnt] = bcast_input_nb[bcast_dim_cnt - 1] *
                                            bcast_input_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt] = bcast_dst_nb[bcast_dim_cnt - 1] *
                                          bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_weight_nb[bcast_dim_cnt] =
                bcast_weight_nb[bcast_dim_cnt - 1] *
                bcast_weight_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}

int addr_malloc(const aclnn_shape_t& shape, void** deviceAddr, size_t size_t, ggml_backend_ascend_context &ctx) {
    auto size = GetShapeSize(shape) * size_t;
    if(size <= 0){
      return 0;
    }
    // 调用aclrtMalloc申请device侧内存
    // auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ggml_ascend_pool_alloc<char> device_allocator(ctx.pool(), size);
    *deviceAddr = static_cast<void *>(device_allocator.get());
    // auto ret = aclrtSynchronizeStream(ctx.stream());
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    return 0;
}

int addr_malloc(const aclnn_shape_t& shape, void** deviceAddr, size_t size_t) {
    auto size = GetShapeSize(shape) * size_t;
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    return 0;
}