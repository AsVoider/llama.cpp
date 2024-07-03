#ifndef ACLNNUNARY_H
#define ACLNNUNARY_H

#include <cstdint>
#include <vector>

int aclnnSigmoidFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnHardSigmoidFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnGeluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnSiluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnTanhFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnReluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);

int aclnnHardsWishFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);
#endif