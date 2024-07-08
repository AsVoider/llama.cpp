#ifndef ACLNNCOMP_H
#define ACLNNCOMP_H

#include <cstdint>
#include <vector>

int aclnnSoftMaxFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream );

void aclnnSoftMaxTest();

#endif