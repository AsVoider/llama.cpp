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

// ne1 = dst->src[0]->ne
// ne2 = dst->src[1]->ne
// data = dst->src[0]->data
// dst = dst->src[1]->data
void aclnnCpyCompute(int64_t* ne1, int64_t* ne2,float* data, float* dst, aclrtContext &context, aclrtStream &stream);


// ne1 = dst->src[0]->ne
// ne2 = dst->src[1]->ne
// ne = dst->ne
// src1 = dst->src[0]->data
// src2 = dst->src[1]->data
// dst = dst->data
void aclnnGetRowsCompute(int64_t* ne1, int64_t* ne2, int64_t* ne, float* src1, float* src2, float* dst ,aclrtContext &context, aclrtStream &stream);

// ne1 = dst->src[0]->ne
// ne2 = dst->src[1]->ne
// ne = dst->ne
// data1 = dst->src[0]->data
// data2 = dst->src[1]->data
// dst = dst->data
void aclnnMulMatCompute(int64_t* ne1, int64_t* ne2, int64_t* ne, float* data1, float* data2, float* dst, aclrtContext &context, aclrtStream &stream);

// ne = dst->src[0]->ne
// x = dst->src[0]->data
// dst = dst->data
// freq_scale = dst->op_params[6];
// freq_base = dst->op_params[5];
// n_dims = dst->op_params[1];
// pos = dst->src[1]->data
void aclnnRopeCompute(int64_t *ne, float freq_scale, float freq_base, int n_dims, int32_t* pos, float* x, float* dst ,aclrtContext &context, aclrtStream &stream);

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