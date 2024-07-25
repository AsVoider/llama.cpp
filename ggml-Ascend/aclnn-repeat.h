#ifndef ACLNNREPEAT_H
#define ACLNNREPEAT_H

#include <cstdint>
#include "common.h"

int aclnn_repeat_func(void* selfDataAddr, void* outDataAddr,
    aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
    aclDataType selfDataType, aclDataType outDataType,
    vector<int64_t>& repeatsArray, aclrtStream &stream);

#endif