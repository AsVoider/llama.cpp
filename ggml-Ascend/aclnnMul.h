//
// Created by 35763 on 2024/6/26.
//

#ifndef ACLNNMUL_CPP_H
#define ACLNNMUL_CPP_H


int aclnnMulFunc(std::vector<int64_t>& selfShape, std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape, std::vector<float>& selfHostData, std::vector<float>& otherHostData, std::vector<float>& outHostData, float* dst);
void aclnnMulTest();

int aclnnMulsFunc(std::vector<float>& selfHostData, float otherValue, int len, int width);
void aclnnMulsTest();

int aclnnMulMatFunc(std::vector<float>& selfHostData, std::vector<float>& mat2HostData, std::vector<int64_t>& selfShape, std::vector<int64_t>& mat2Shape,std::vector<int64_t>& outShape, std::vector<float>& outHostData);
void aclnnMulMatTest();





#endif //ACLNNMUL_CPP_H
