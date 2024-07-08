#include <iostream>
#include <cstring>
#include "common.h"
#include "acl/acl.h"
#include "aclnn-operator.h"
#include "aclnn-sqrt.h"
#include "aclnn-leaky.h"
#include "aclnn-mul.h"
#include "aclnn-unary.h"
#include "aclnn-clamp.h"
#include "aclnn-add.h"
#include "aclnn-norm.h"
#include "aclnn-cpy.h"
#include "aclnn-comp.h"

void aclnnSiluCompute(int64_t* ne, float* data, float* dst, aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnSiluFunc(selfShape, outShape, selfHostData, outHostData, dst, context, stream);
}

void aclnnSoftMaxCompute(int64_t* ne, float* data, float* dst, float scale, aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> selfShape = {ne[0], ne[1], ne[2], ne[3]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData(selfHostData.size(), 0);
  float dstTemp[ne[3]*ne[2]*ne[1]*ne[0]];
  int ret = aclnnMulsFunc(selfHostData, outHostData, scale, selfShape , outShape, dstTemp, context, stream);
  std::vector<int64_t> selfShape1 = {ne[0], ne[1], ne[2], ne[3]};
  std::vector<int64_t> outShape1 = selfShape1;
  std::vector<float> selfHostData1(dstTemp, dstTemp + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> outHostData1(selfHostData1.size(), 0);
  int ret1 = aclnnSoftMaxFunc(selfShape1, outShape1, selfHostData1, outHostData1, dst, context, stream);
}

void aclnnRmsNormCompute(int64_t* ne, float* data, float* dst, float epi, aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> xShape =  {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> gammaShape = {ne[0]};
  std::vector<int64_t> yShape = xShape;
  std::vector<int64_t> rstdShape = {ne[0], 1};
  std::vector<float> xHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]);
  std::vector<float> gammaHostData(ne[0], 1);
  std::vector<float> yHostData(xHostData.size(), 0);
  std::vector<float> rstdHostData = {1, float(ne[0])};
  int ret = aclnnRmsNormFunc(xShape, gammaShape, yShape, rstdShape, xHostData, gammaHostData, yHostData, rstdHostData, epi, dst, context, stream);
}

void aclnnAddCompute(int64_t* ne1, int64_t* ne2,float* data1, float* data2, float* dst, aclrtContext &context, aclrtStream &stream){
  std::vector<float> selfHostData(data1, data1 + ne1[3]*ne1[2]*ne1[1]*ne1[0]);
  std::vector<float> otherHostData(data2, data2 + ne2[3]*ne2[2]*ne2[1]*ne2[0]);
  float alphaValue = 1.0f;
  std::vector<int64_t> selfShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
  std::vector<int64_t> otherShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
  std::vector<int64_t> outShape = {
    (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
    (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
    (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
    (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
  };
  int ret = aclnnAddFunc(selfHostData, otherHostData,alphaValue, selfShape, otherShape, outShape, dst, context, stream);
}

void aclnnMulsCompute(int64_t* ne1, int64_t* ne2,float* data1, float* data2, float* dst, aclrtContext &context, aclrtStream &stream){
  std::vector<float> selfHostData(data1, data1 + ne1[3]*ne1[2]*ne1[1]*ne1[0]);
  std::vector<float> otherHostData(data2, data2 + ne2[3]*ne2[2]*ne2[1]*ne2[0]);
  float alphaValue = 1.0f;
  std::vector<int64_t> selfShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
  std::vector<int64_t> otherShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
  std::vector<int64_t> outShape = {
    (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
    (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
    (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
    (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
  };
  std::vector<float> outHostData(outShape[0]*outShape[1]*outShape[2]*outShape[3], 0);
  int ret = aclnnMulFunc(selfShape, otherShape, outShape, selfHostData, otherHostData, outHostData, dst, context, stream);
}

void aclnnCpyCompute(int64_t* ne1, int64_t* ne2,float* data, float* dst, aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> srcShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
  std::vector<int64_t> selfRefShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
  std::vector<float> selfRefHostData(dst, dst + ne2[3]*ne2[2]*ne2[1]*ne2[0]);
  std::vector<float> srcHostData(data, data+ ne1[3]*ne1[2]*ne1[1]*ne1[0]);
  int ret = aclnnCpyFunc(selfRefShape, srcShape, selfRefHostData, srcHostData, dst, context, stream);
}

void aclnnGetRowsCompute(float* dst ,aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {2};
  std::vector<int64_t> outShape = {2, 2};
  int64_t dim = 0;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> indexHostData = {1, 0};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int ret = aclnnGetRowsFunc(selfShape, indexShape, outShape, dim, selfHostData, indexHostData, outHostData, dst, context, stream);
}

///not for usage today 



void aclnnSqrtCompute(int64_t* ne, float* data, float* dst){
  std::vector<int64_t> selfShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<int64_t> outShape = selfShape;
  std::vector<float> selfHostData(data, data + ne[3]*ne[2]*ne[1]*ne[0]) ;
  std::vector<float> outHostData(selfHostData.size(), 0);
  int ret = aclnnSqrtFunc(selfShape, outShape, selfHostData, outHostData, dst);
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