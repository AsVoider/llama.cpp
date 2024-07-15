#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
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
#include "aclnn-rope.h"
#include "aclnn-math.h"
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

void aclnnMulCompute(int64_t* ne1, int64_t* ne2,float* data1, float* data2, float* dst, aclrtContext &context, aclrtStream &stream){
  std::vector<float> selfHostData(data1, data1 + ne1[3]*ne1[2]*ne1[1]*ne1[0]);
  std::vector<float> otherHostData(data2, data2 + ne2[3]*ne2[2]*ne2[1]*ne2[0]);
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

void aclnnGetRowsCompute(int64_t* ne1, int64_t* ne2, int64_t* ne, float* src1, float* src2, float* dst ,aclrtContext &context, aclrtStream &stream){
  std::vector<int64_t> selfShape = {1, 1, ne1[1]*ne1[2]*ne1[3], ne1[0]};
  std::vector<int64_t> indexShape = {ne2[0]*ne2[1]*ne2[2]};
  std::vector<int64_t> outShape = {1, 1, ne2[0]*ne2[1]*ne2[2], ne1[0]};
  int64_t dim = 2;
  std::vector<float> selfHostData(src1 ,src1+ ne1[0]*ne1[1]*ne1[2]*ne1[3]);
  std::vector<int> indexHostData(src2, src2+ ne2[0]*ne2[1]*ne2[2]*ne2[3]);
  std::vector<float> outHostData(ne[0]*ne[1]*ne[2]*ne[3], 0);
  for(int i = 0; i < ne2[1]*ne2[2]*ne2[3]; i++){
    for(int j = 0; j< ne2[0]; j++){
      indexHostData[i*ne2[0]+j] += i* ne1[1];
    }
  }
  for(int i = 0; i< indexHostData.size();i++){
    std::cout<< indexHostData[i]<<" ";
  }
  std::cout<<std::endl;
  int ret = aclnnGetRowsFunc(selfShape, indexShape, outShape, dim, selfHostData, indexHostData, outHostData, dst, context, stream);
}

void aclnnMulMatCompute(int64_t* ne1, int64_t* ne2, int64_t* ne, float* data1, float* data2, float* dst, aclrtContext &context, aclrtStream &stream){
  std::vector<float> selfHostData(data1, data1 + ne1[3]*ne1[2]*ne1[1]*ne1[0]);
  std::vector<float> otherHostData(data2, data2 + ne2[3]*ne2[2]*ne2[1]*ne2[0]);
  std::vector<int64_t> selfShape = {ne1[3], ne1[2], ne1[1], ne1[0]};
  std::vector<int64_t> otherShape = {ne2[3], ne2[2], ne2[1], ne2[0]};
  std::vector<int64_t> outShape = {ne[3], ne[2], ne[1], ne[0]};
  std::vector<float> outHostData(outShape[0]*outShape[1]*outShape[2]*outShape[3], 0);
  int ret = aclnnMulMatFunc(selfHostData, otherHostData, selfShape, otherShape, outShape, outHostData, dst, context, stream);
}

void aclnnRopeCompute(int64_t *ne, float freq_scale, float freq_base, int n_dims, int32_t* pos, float* x, float* dst ,aclrtContext &context, aclrtStream &stream){
  int64_t size = ne[0] *ne[1] *ne[2] *ne[3];
  float theta_scale_pow[ne[0]/2];
  float theta_base[size];
  float theta[size];
  float sin_d[size];
  float cos_d[size];

  float theta_scale = pow(freq_base, -2.0/n_dims);
  // std::cout<< theta_scale <<std::endl;
  // return;

  std::vector<float> powExpHostData(ne[0]/2, 0);
  std::vector<float> powOutData = powExpHostData;
  std::vector<int64_t> powExpShape{ne[0]/2, 1};
  std::vector<int64_t> powOutShape = powExpShape;
  std::generate_n(powExpHostData.begin(), powExpHostData.size(), [&, index = 0]() mutable {
    return index++;
  });

  // for (int i=0; i< powExpHostData.size(); i++){
  //   std::cout << powExpHostData[i]<< " "; 
  // }
  // return;

  int ret = aclnnPowScalarTensorFunc(powExpShape, powOutShape, powExpHostData, powOutData, theta_scale, theta_scale_pow, context, stream);
  std::vector<float> mulOtherHostData(size, 0);
  std::generate_n(mulOtherHostData.begin(), size, [&, index = 0]() mutable {
    return theta_scale_pow[index++ % (ne[0]/2)];
  });
  // for (int i=0; i< mulOtherHostData.size(); i++){
  //   std::cout << mulOtherHostData[i]<< " "; 
  // }
  // return;

  std::vector<int64_t> mulSelfShape{size, 1};
  std::vector<int64_t> mulOtherShape = mulSelfShape;
  std::vector<int64_t> mulOutShape = mulSelfShape;

  std::vector<float> mulOutHostData(size, 0);
  std::vector<float> mulSelfHostData(size, 0);
  std::generate_n(mulSelfHostData.begin(), size, [&, index = 0]() mutable {
    return (float)pos[index++ % ne[2]];
  });

  // for (int i=0; i< mulSelfHostData.size(); i++){
  //   std::cout << mulSelfHostData[i]<< " "; 
  // }
  // return;

  ret = aclnnMulFunc(mulSelfShape, mulOtherShape, mulOutShape, mulSelfHostData, mulOtherHostData, mulOutHostData, theta_base, context, stream);

  std::vector<int64_t> mulsSelfShape = mulSelfShape;
  std::vector<int64_t> mulsOutShape = mulOutShape;
  std::vector<float> mulsOutHostData(size, 0);
  std::vector<float> mulsSelfHostData(theta_base, theta_base+ size);

  // for (int i=0; i< mulsSelfHostData.size(); i++){
  //   std::cout << mulsSelfHostData[i]<< " "; 
  // }
  // return;

  ret = aclnnMulsFunc(mulsSelfHostData, mulsOutHostData, freq_scale, mulsSelfShape, mulsOutShape, theta, context, stream);


  std::vector<int64_t> sinSelfShape = {size, 1};
  std::vector<int64_t> sinShape = {size, 1};
  std::vector<float> sinSelfHostData(theta, theta+ size);
  std::vector<float> sinHostData(size, 0);

  // for (int i=0; i< sinSelfHostData.size(); i++){
  //   std::cout << sinSelfHostData[i]<< " "; 
  // }
  // return;


  ret = aclnnSinFunc(sinSelfShape, sinShape, sinSelfHostData, sinHostData, sin_d, context, stream);

  std::vector<int64_t> cosSelfShape = {size, 1};
  std::vector<int64_t> cosShape = {size, 1};
  std::vector<float> cosSelfHostData(theta, theta+ size);
  std::vector<float> cosHostData(size, 0);
  ret = aclnnCosFunc(cosSelfShape, cosShape, cosSelfHostData, cosHostData, cos_d, context, stream);

  std::vector<int64_t> queryShape = {1, size/ne[0], 1, ne[0]};
  std::vector<int64_t> keyShape = queryShape;
  std::vector<int64_t> sinShapeRp = {1, size/ne[0], 1, ne[0]};
  std::vector<int64_t> cosShapeRp = sinShapeRp;
  std::vector<float> queryHostData(x, x+size);
  std::vector<float> keyHostData(size, 0);
  std::vector<float> sinQKHostData(sin_d, sin_d +size);
  std::vector<float> cosQKHostData(cos_d, cos_d +size);
  ret = aclnnRoPEFunc(queryShape, keyShape, cosShapeRp, sinShapeRp, queryHostData, keyHostData, cosQKHostData, sinQKHostData, dst, context, stream); 
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