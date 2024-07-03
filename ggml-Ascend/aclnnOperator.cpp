#include <iostream>
#include "aclnnOperator.h"
#include "aclnnSqrt.h"
#include "aclnnLeaky.h"
#include "aclnnMul.h"
#include "aclnnUnary.h"
#include "aclnnClamp.h"

void aclnnSqrtCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]) ;
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnSqrtFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnSqrCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> otherShape = selfShape;
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> otherHostData = selfHostData;
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnMulFunc(selfShape, otherShape, outShape, selfHostData, otherHostData, outHostData, dst);
}

void aclnnLeakyReluCompute(int64_t* ne, float* data, float* dst, float negativeSlopeValue){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnLeakyReluFunc(selfShape, outShape, selfHostData, outHostData, negativeSlopeValue, dst);
}

void aclnnSigmoidCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnSigmoidFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnHardSigmoidCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnHardSigmoidFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnGeluCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnGeluFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnSiluCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnSiluFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnTanhCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnTanhFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnReluCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnReluFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnHardsWishCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnHardsWishFunc(selfShape, outShape, selfHostData, outHostData, dst);
}

void aclnnClampCompute(int64_t* ne, float* data, int32_t* op_params, float* dst){
  std::vector<int64_t> shape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  float min;
  float max;
  memcpy(&min, op_params, sizeof(float));
  memcpy(&max, (float *) op_params + 1, sizeof(float));
  std::cout << min << " " << max << std::endl;
  int ret = aclnnClampFunc(shape, selfHostData, outHostData, max, min);
}