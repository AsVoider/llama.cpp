#ifndef ACLNNCOMP_H
#define ACLNNCOMP_H

#include <cstdint>
#include <vector>

int aclnnSoftMaxFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream );

int acl_soft_max_func(void* selfDataAddr, void* outDataAddr, aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
  aclDataType selfDataType, aclDataType otherDataType, aclDataType outDataType,
  aclrtStream &stream);

void aclnnSoftMaxTest();

#endif