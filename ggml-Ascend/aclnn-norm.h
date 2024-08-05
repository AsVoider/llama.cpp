#ifndef NORM_H
#define NORM_H

#include <iostream>
#include "common.h"
#include <vector>


int aclnnNormFunc(std::vector<float>& selfHostData, std::vector<float>& outHostData, std::vector<int64_t>& dimData, float pValue, bool keepDim, std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape);
void aclnnNormTest();

int aclnnGroupNormFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& gammaShape, std::vector<int64_t>& betaShape, std::vector<int64_t>& outShape,  std::vector<int64_t>& meanOutShape, 
        std::vector<int64_t>& rstdOutShape, std::vector<float>& selfHostData, std::vector<float>& gammaHostData, std::vector<float>& betaHostData, std::vector<float>& outHostData, 
        std::vector<float>& meanOutHostData, std::vector<float>& rstdOutHostData, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps);
void aclnnGroupNormTest();

int aclnn_rms_norm_func(void* xDataAddr, void* gammaDataAddr, void* yDataAddr, void* rstdDataAddr,
  aclnn_shape_t& xShape, aclnn_shape_t& gammaShape, aclnn_shape_t& yShape, aclnn_shape_t& rstdShape,
  aclDataType xDataType, aclDataType gammaDataType, aclDataType yDataType, aclDataType rstdDataType,
  float epsilon, aclrtStream &stream);

int aclnnRmsNormFunc( std::vector<int64_t>& xShape, std::vector<int64_t>& gammaShape, std::vector<int64_t>& yShape, std::vector<int64_t>& rstdShape,
  std::vector<float>& xHostData, std::vector<float>& gammaHostData, std::vector<float>& yHostData, std::vector<float>& rstdHostData, float epsilon, float* dst, aclrtContext &context, aclrtStream &stream);

void aclnnRmsNormTest();

#endif