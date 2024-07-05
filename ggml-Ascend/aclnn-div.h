#ifndef DIV_H
#define DIV_H

#include <iostream>
#include <vector>

int aclnnDivFunc(std::vector<int64_t>& selfShape,std::vector<int64_t>& otherShape, std::vector<int64_t>& outShape,std::vector<float>& selfHostData, std::vector<float>& otherHostData, std::vector<float>& outHostData);
void aclnnDivTest();



#endif
