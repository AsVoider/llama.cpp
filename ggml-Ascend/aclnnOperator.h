#ifndef ACLNNOPERATOR_H
#define ACLNNOPERATOR_H

#include <cstdint>
#include <vector>

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
void aclnnSiluCompute(int64_t* ne, float* data, float* dst);

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