#ifndef ACLNNROPE_H
#define ACLNNROPE_H

#include <cstdint>
#include <vector>
#include "common.h"

int aclnn_rope_func(void* queryDataAddr, void* keyDataAddr, void* cosDataAddr, void* sinDataAddr,
    aclnn_shape_t& queryShape, aclnn_shape_t& keyShape, aclnn_shape_t& cosShape, aclnn_shape_t& sinShape,
    aclDataType queryDataType, aclDataType keyDataType, aclDataType cosDataType, aclDataType sinDataType,
    aclrtStream &stream);

int aclnnRoPEFunc(std::vector<int64_t>& queryShape, std::vector<int64_t>& keyShape, std::vector<int64_t>& cosShape, std::vector<int64_t>& sinShape, 
    std::vector<float>& queryHostData, std::vector<float>& keyHostData, std::vector<float>& cosHostData, std::vector<float>& sinHostData, float* dst ,aclrtContext &context, aclrtStream &stream);


void aclnnRoPETest();

#endif