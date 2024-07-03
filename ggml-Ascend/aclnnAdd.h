//
// Created by 35763 on 2024/6/26.
//

#ifndef ACLNNADD_H
#define ACLNNADD_H

#include <cstdint>
#include <vector>

int aclnnAddFunc(std::vector<float>& selfHostData,std::vector<float>& otherHostData, std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape);

void aclnnAddTest();

#endif //ACLNNADD_H
