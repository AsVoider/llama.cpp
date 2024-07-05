#ifndef ACLNNSORT_H
#define ACLNNSORT_H

#include <cstdint>
#include <vector>

int aclnnArgSortFunc(int64_t dim ,bool descending, std::vector<int64_t>& selfShape, std::vector<int64_t>& outIndicesShape, std::vector<int64_t>& selfHostData, std::vector<int64_t>& outIndicesHostData);


void aclnnArgSortTest();


#endif