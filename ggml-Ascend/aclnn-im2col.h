#ifndef ACLNNIM2COL_H
#define ACLNNIM2COL_H

#include <cstdint>
#include <vector>

int aclnnIm2ColFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<int64_t>& kernelSizeData, std::vector<int64_t>& dilationData, std::vector<int64_t>& paddingData, std::vector<int64_t>& strideData, std::vector<float>& outHostData);

void aclnnIm2ColTest();

#endif