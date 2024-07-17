//
// Created by 35763 on 2024/6/26.
//

#ifndef ACLNNMUL_CPP_H
#define ACLNNMUL_CPP_H

#include <cstdint>
#include <vector>
#include "common.h"

int aclnn_mul_func(void* selfDataAddr, void* otherDataAddr, void* outDataAddr,
    aclnn_shape_t& selfShape, aclnn_shape_t& otherShape, aclnn_shape_t& outShape,
    aclDataType selfDataType, aclDataType otherDataType, aclDataType outDataType,
    aclrtStream &stream);
void aclnn_mul_func_test(int64_t* ne1, int64_t* ne2, float* data1, float* data2, int32_t deviceId, aclrtStream stream);


int aclnnMulFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& otherHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream);
void aclnnMulTest();

int aclnnMulsFunc(std::vector<float>& selfHostData, std::vector<float>& outHostData, float otherValue, std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape,float* dst, aclrtContext &context, aclrtStream &stream);
void aclnnMulsTest();

int aclnnMulMatFunc(std::vector<float>& selfHostData, std::vector<float>& mat2HostData, std::vector<int64_t>& selfShape, std::vector<int64_t>& mat2Shape,std::vector<int64_t>& outShape, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream);
void aclnnMulMatTest();





#endif //ACLNNMUL_CPP_H
