#ifndef CPY_H
#define CPY_H

#include <iostream>
#include <vector>
#include "acl/acl.h"

int aclnnCpyFunc(std::vector<int64_t>& selfRefShape, std::vector<int64_t>& srcShape, std::vector<float>& selfRefHostData, std::vector<float>& srcHostData,  float* dst, aclrtContext &context, aclrtStream &stream);

int aclnnGetRowsFunc(  std::vector<int64_t> selfShape,
  std::vector<int64_t> indexShape,
  std::vector<int64_t> outShape,
  int64_t dim,
  std::vector<float> selfHostData,
  std::vector<int> indexHostData,
  std::vector<float> outHostData,float* dst, aclrtContext &context, aclrtStream &stream);

#endif