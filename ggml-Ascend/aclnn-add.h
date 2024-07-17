//
// Created by 35763 on 2024/6/26.
//

#ifndef ACLNNADD_H
#define ACLNNADD_H

#include <cstdint>
#include <vector>
#include "common.h"

int aclnn_add_func(void* selfDataAddr, void* otherDataAddr, void* outDataAddr,
	aclnn_shape_t& selfShape, aclnn_shape_t& otherShape, aclnn_shape_t& outShape,
    aclDataType selfDataType, aclDataType otherDataType, aclDataType outDataType,
    aclrtStream &stream);

int aclnnAddFunc(std::vector<float>& selfHostData,std::vector<float>& otherHostData, float alphaValue, std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,float* dst, aclrtContext &context, aclrtStream &stream);

void aclnnAddTest();

void aclnn_add_func_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream);

#endif //ACLNNADD_H
