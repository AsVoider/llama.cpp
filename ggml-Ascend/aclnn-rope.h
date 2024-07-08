#ifndef ACLNNROPE_H
#define ACLNNROPE_H

#include <cstdint>
#include <vector>

int aclnnRoPEFunc(std::vector<int64_t>& queryShape, std::vector<int64_t>& keyShape, std::vector<int64_t>& cosShape, std::vector<int64_t>& sinShape, 
    std::vector<float>& queryHostData, std::vector<float>& keyHostData, std::vector<float>& cosHostData, std::vector<float>& sinHostData );

void aclnnRoPETest();

#endif