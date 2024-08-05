#ifndef ACLNN_MATH_H
#define ACLNN_MATH_H

#include <iostream>
#include <vector>
#include "common.h"

template <typename T>
int aclnn_pow_scalar_tensor_func(T selfValue, void* exponentDataAddr, void* outDataAddr,
  aclnn_shape_t& exponentShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType exponentDataType, aclDataType outDataType,
  aclrtStream &stream);

int aclnnPowScalarTensorFunc(std::vector<int64_t>& exponentShape, std::vector<int64_t> &outShape,
  std::vector<float> &exponentHostData, std::vector<float>& outHostData, float selfValue, float* dst ,aclrtContext &context, aclrtStream &stream);

int aclnn_sin_func(void* selfDataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType outDataType,
  aclrtStream &stream);

int aclnnSinFunc(std::vector<int64_t> &selfShape,
  std::vector<int64_t> &outShape,
  std::vector<float> &selfHostData,
  std::vector<float> &outHostData, float* dst ,aclrtContext &context, aclrtStream &stream);

int aclnn_cos_func(void* selfDataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType outDataType,
  aclrtStream &stream);

int aclnnCosFunc( std::vector<int64_t> &selfShape,
  std::vector<int64_t> &outShape,
  std::vector<float> &selfHostData,
  std::vector<float> &outHostData, float* dst ,aclrtContext &context, aclrtStream &stream);

#endif