#ifndef ACLNNCOMP_H
#define ACLNNCOMP_H

#include <cstdint>
#include <vector>
#include "common.h"

int aclnnSoftMaxFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream );

int aclnn_soft_max_func(void* selfDataAddr, void* outDataAddr,
  aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType outDataType,
  aclrtStream &stream);

int acl_soft_max_func(void* selfDataAddr, void* outDataAddr, aclnn_shape_t& selfShape, 
  aclnn_shape_t& outShape, aclDataType selfDataType, 
  aclDataType otherDataType, aclDataType outDataType,aclrtStream &stream);

int aclnn_soft_max_func(void * dataAddr, void * maskAddr, float scale, void * outAddr,
    aclnn_shape_t & dataShape, aclnn_shape_t & maskShape, aclnn_shape_t & outShape,
    aclDataType dataType, aclDataType maskType, aclDataType outType, ggml_backend_ascend_context & ctx);

void aclnnSoftMaxTest();

#endif