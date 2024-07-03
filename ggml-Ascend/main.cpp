#include <iostream>
#include <chrono>
#include <vector>
#include "acl/acl.h"
#include "common.h"
#include "aclnnOperator.h"
#include "aclnnAdd.h"
#include "aclnnMul.h"
#include "aclnnNorm.h"
#include "aclnnDiv.h"
#include "aclnnSort.h"
#include "aclnnComp.h"
#include "aclnnArange.h"
#include "aclnnClamp.h"
#include "aclnnLeaky.h"
#include "aclnnIm2Col.h"
#include "aclnnRoPE.h"
#include "aclnnSqrt.h"


int main() {

  int64_t ne[4] = {2, 1, 2, 2};
  float a[24] = {0.0, -1.0, 2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1.3, 2.5, 6.7, -4, -1.4, -1.6, -8, -16.9};
  float b[24];
  float pa[5] = {1.0, 8.0};
  auto start = std::chrono::high_resolution_clock::now();

  aclnnClampTest();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "running time: " << duration.count() << " ms" << std::endl;

  aclFloat16 A = 1.0;
  std::cout << "Elements of array b:" << std::endl;
  for (int i = 0; i < 24; ++i) {
    std::cout << b[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}

