#ifndef ACLNNLEAKY_H
#define ACLNNLEAKY_H

#include <cstdint>
#include <vector>

int aclnnLeakyReluFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float negativeSlopeValue, float* dst);

void aclnnLeakyReluTest();

#endif