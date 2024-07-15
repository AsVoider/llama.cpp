#ifndef ACLNNUNARY_H
#define ACLNNUNARY_H

#include <cstdint>
#include <vector>
#include "common.h"

int aclnn_silu_func(void* selfDataAddr, void* outDataAddr,
    aclnn_shape_t selfShape, aclnn_shape_t outShape,
    aclDataType selfDataType, aclDataType outDataType,
    aclrtStream &stream);

int aclnnSigmoidFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnHardSigmoidFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnGeluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnSiluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream);

int aclnnTanhFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnReluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnHardsWishFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);
#endif