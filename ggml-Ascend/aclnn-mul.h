//
// Created by 35763 on 2024/6/26.
//

#ifndef ACLNNMUL_CPP_H
#define ACLNNMUL_CPP_H

#include <cstdint>
#include <vector>

int aclnnMulFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& otherHostData, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream);
void aclnnMulTest();

int aclnnMulsFunc(std::vector<float>& selfHostData, std::vector<float>& outHostData, float otherValue, std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape,float* dst, aclrtContext &context, aclrtStream &stream);
void aclnnMulsTest();

int aclnnMulMatFunc(std::vector<float>& selfHostData, std::vector<float>& mat2HostData, std::vector<int64_t>& selfShape, std::vector<int64_t>& mat2Shape,std::vector<int64_t>& outShape, std::vector<float>& outHostData, float* dst, aclrtContext &context, aclrtStream &stream);
void aclnnMulMatTest();





#endif //ACLNNMUL_CPP_H
