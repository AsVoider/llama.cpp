#ifndef ACLNNSQRT_H
#define ACLNNSQRT_H

#include <cstdint>
#include <vector>

int aclnnSqrtFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst);




#endif