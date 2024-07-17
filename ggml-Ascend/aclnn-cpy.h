#ifndef CPY_H
#define CPY_H

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "common.h"

int aclnn_cpy_func(void* selfRefDataAddr, void* srcDataAddr,
    aclnn_shape_t& selfRefShape, aclnn_shape_t& srcShape,
    aclDataType selfRefDataType, aclDataType srcDataType,
    aclrtStream &stream);

int aclnnCpyFunc(std::vector<int64_t>& selfRefShape, std::vector<int64_t>& srcShape, std::vector<float>& selfRefHostData, std::vector<float>& srcHostData,  float* dst, aclrtContext &context, aclrtStream &stream);

int aclnn_get_rows_func(void* selfDataAddr, void* indexDataAddr, void* outDataAddr,
	aclnn_shape_t& selfShape, aclnn_shape_t& indexShape, aclnn_shape_t& outShape,
	aclDataType selfDataType, aclDataType indexDataType, aclDataType outDataType,
	aclrtStream &stream);

int aclnnGetRowsFunc(std::vector<int64_t> &selfShape,
  std::vector<int64_t> &indexShape,
  std::vector<int64_t> &outShape,
  int64_t dim,
  std::vector<float> &selfHostData,
  std::vector<int> &indexHostData,
  std::vector<float> &outHostData,float* dst, aclrtContext &context, aclrtStream &stream);

void aclnn_cpy_func_test(int64_t* ne, float* data, int32_t deviceId, aclrtStream stream);

#endif