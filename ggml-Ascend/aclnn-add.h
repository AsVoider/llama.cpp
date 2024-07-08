//
// Created by 35763 on 2024/6/26.
//

#ifndef ACLNNADD_H
#define ACLNNADD_H

#include <cstdint>
#include <vector>
#include "common.h"

int aclnnAddFunc(std::vector<float>& selfHostData,std::vector<float>& otherHostData, float alphaValue, std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,float* dst, aclrtContext &context, aclrtStream &stream);

void aclnnAddTest();

#endif //ACLNNADD_H
