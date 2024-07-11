#ifndef ACLNN_MATH_H
#define ACLNN_MATH_H

#include <iostream>
#include <vector>
#include "common.h"


int aclnnPowScalarTensorFunc(std::vector<int64_t>& exponentShape, std::vector<int64_t> &outShape,
  std::vector<float> &exponentHostData, std::vector<float>& outHostData, float selfValue, float* dst ,aclrtContext &context, aclrtStream &stream);

int aclnnSinFunc(std::vector<int64_t> &selfShape,
  std::vector<int64_t> &outShape,
  std::vector<float> &selfHostData,
  std::vector<float> &outHostData, float* dst ,aclrtContext &context, aclrtStream &stream);

int aclnnCosFunc( std::vector<int64_t> &selfShape,
  std::vector<int64_t> &outShape,
  std::vector<float> &selfHostData,
  std::vector<float> &outHostData, float* dst ,aclrtContext &context, aclrtStream &stream);

#endif