#ifndef ACLNNPERMUTE_H
#define ACLNNPERMUTE_H

#include <cstdint>
#include "common.h"

int aclnn_permute_func(void* selfDataAddr, void* outDataAddr,
    aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
    aclDataType selfDataType, aclDataType outDataType,
    aclnn_shape_t& dimsData, aclrtStream &stream);

#endif