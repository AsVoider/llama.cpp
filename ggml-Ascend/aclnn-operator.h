#ifndef ACLNNOPERATOR_H
#define ACLNNOPERATOR_H

#include <cstdint>
#include <vector>



// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnSiluCompute(int64_t* ne, float* data, float* dst, aclrtContext &context, aclrtStream &stream);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
// scale = dst->op_params[0]
void aclnnSoftMaxCompute(int64_t* ne, float* data, float* dst, float scale, aclrtContext &context, aclrtStream &stream);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
// epi = dst->op_params
void aclnnRmsNormCompute(int64_t* ne, float* data, float* dst, float epi, aclrtContext &context, aclrtStream &stream);

// ne1 = dst->src[0]->ne
// ne2 = dst->src[1]->ne
// data1 = dst->src[0]->data
// data2 = dst->src[1]->data
// dst = dst->data
void aclnnAddCompute(int64_t* ne1, int64_t* ne2, float* data1, float* data2, float* dst, aclrtContext &context, aclrtStream &stream);

/*
    Not For Usage Operators :
*/

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnSqrtCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnSqrCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
// negativeSlopeValue = dst->op_params
void aclnnLeakyReluCompute(int64_t* ne, float* data, float* dst, float negativeSlopeValue);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnSigmoidCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnHardSigmoidCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnGeluCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnTanhCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnReluCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// dst = dst->data
void aclnnHardsWishCompute(int64_t* ne, float* data, float* dst);

// ne = dst->src[0]->ne
// data = dst->src[0]->data
// op_params = dst->op_params
// dst = dst->data
void aclnnClampCompute(int64_t* ne, float* data, int32_t* op_params, float* dst);

#endif