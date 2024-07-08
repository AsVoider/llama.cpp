#ifndef ACLNNCLAMP_H
#define ACLNNCLAMP_H

#include <cstdint>
#include <vector>

int aclnnClampFunc(std::vector<int64_t>& shape, std::vector<float>& selfHostData, std::vector<float>& outHostData, float max_v, float min_v );

void aclnnClampTest();

#endif