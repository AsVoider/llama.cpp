#include "aclnn-compute.h"
#include "aclnn-comp.h"
#include "string.h"
#include "aclnn-add.h"
#include "aclnn-mul.h"
#include "aclnn-cpy.h"
#include "aclnn-unary.h"
#include "aclnn-comp.h"
#include "aclnn-norm.h"
#include "aclnn-math.h"
#include "aclnn-rope.h"
#include "aclnn-permute.h"
#include "aclnn-repeat.h"
#include <cstring>
#include <cmath>
#include <algorithm>

static aclTensor* aclnn_zero(ggml_backend_ascend_context & ctx, void* buffer,
                             size_t n_bytes, int64_t* ne, int64_t dims,
                             aclDataType type, size_t type_size) {
    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_size;
    for (int i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    aclrtMemset(buffer, n_bytes, 0, n_bytes);
    aclTensor* zero =
        ggml_ascend_create_tensor(buffer, type, type_size, ne, nb, dims);
    return zero;
}

static aclTensor* aclnn_ones(ggml_backend_ascend_context& ctx, void* buffer,
                             size_t n_bytes, int64_t* ne, int64_t dims,
                             aclDataType type, size_t type_size,
                             float value = 1.0f) {
    aclTensor* acl_tensor =
        aclnn_zero(ctx, buffer, n_bytes, ne, dims, type, type_size);
    float alpha_host = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_host, aclDataType::ACL_FLOAT);
    aclScalar* other = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnInplaceAddsGetWorkspaceSize(acl_tensor, other, alpha,
                                               &workspaceSize, &executor);

    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclrtFree(workspaceAddr);
    return acl_tensor;
}

static void aclnn_fill_scalar(ggml_backend_ascend_context& ctx, float scalar,
                              aclTensor* acl_dst) {
    auto acl_scalar = aclCreateScalar(&scalar, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnInplaceFillScalarGetWorkspaceSize(
        acl_dst, acl_scalar, &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnInplaceFillScalar(workspaceAddr, workspaceSize, executor,
                                     ctx.stream());
    aclDestroyScalar(acl_scalar);
    aclrtFree(workspaceAddr);
}

static void aclnn_arange(ggml_backend_ascend_context& ctx, aclTensor* acl_dst,
                         float start, float stop, float step,
                         int64_t n_elements) {
    int64_t steps = (int64_t)std::ceil((stop - start) / step);
    GGML_ASSERT(n_elements == steps);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_FLOAT);
    aclScalar* acl_step = aclCreateScalar(&step, aclDataType::ACL_FLOAT);

    aclnnArangeGetWorkspaceSize(acl_start, acl_end, acl_step, acl_dst,
                                          &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnArange(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclDestroyScalar(acl_start);
    aclDestroyScalar(acl_end);
    aclDestroyScalar(acl_step);
    aclrtFree(workspaceAddr);
}

static void aclnn_mul(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_other, aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnMulGetWorkspaceSize(acl_src, acl_other, acl_dst,
                                       &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnMul(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclrtFree(workspaceAddr);
}

static void aclnn_inplace_mul(ggml_backend_ascend_context& ctx,
                              aclTensor* acl_src, aclTensor* acl_other) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnInplaceMulGetWorkspaceSize(acl_src, acl_other,
                                              &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    
    aclnnInplaceMul(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclrtFree(workspaceAddr);
}

void ggml_ascend_get_rows(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    // FILE * f = fopen("get_rows_npu_bamboo.txt", "a");
    // fprintf(f, "get_rows_npu\n");
    // fprintf(f, "src0 type: %d\n", src0->type);
    // fprintf(f, "src1 type: %d\n", src1->type);
    // fprintf(f, "dst type: %d\n", dst->type);
    // fprintf(f, "src0 ne: %ld %ld %ld %ld\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    // fprintf(f, "src1 ne: %ld %ld %ld %ld\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    // fprintf(f, "dst ne: %ld %ld %ld %ld\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    // fclose(f);

    aclnn_shape_t selfShape = {1, 1, ne01 * ne02 * ne03, ne00};
    aclnn_shape_t indexShape = {ne10 * ne11 * ne12};
    aclnn_shape_t outShape = {1, 1, ne10 * ne11 * ne12, ne00};
//     std::vector<int64_t> tmpHostData(ne10 * ne11 * ne12 * ne13, 0);
//     std::vector<int64_t> offset;
//     for(int i = 0; i < ne11 * ne12 * ne13; i++) {
//         offset.insert(offset.end(), ne10, i * ne01);
//     }
// 
//     void* offsetDeviceAddr = nullptr;
//     void* tmpDeviceAddr = nullptr;
//     
//     int ret = data_addr_malloc(indexShape, offset, &offsetDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);
// 
//     ret = data_addr_malloc(indexShape, tmpHostData, &tmpDeviceAddr);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);
// 
//     ret = aclnn_add_func(src1->data, offsetDeviceAddr, tmpDeviceAddr,
//                         indexShape, indexShape, indexShape,
//                         ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type],
//                         stream);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);

    auto ret = aclnn_get_rows_func(src0->data, src1->data, dst->data,
                            selfShape, indexShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_get_rows failed. ERROR: %d\n", ret); return);

    // aclrtFree(offsetDeviceAddr);
    // aclrtFree(tmpDeviceAddr);
}

void ggml_ascend_add(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    aclnn_shape_t selfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t otherShape = {ne13, ne12, ne11, ne10};
    aclnn_shape_t outShape = {
        (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
        (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
        (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
        (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
    };

    int ret = aclnn_add_func(src0->data, src1->data, dst->data,
                            selfShape, otherShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_add failed. ERROR: %d\n", ret); return);
}

void ggml_ascend_add_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    auto src0(dst->src[0]), src1(dst->src[1]);
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    aclTensor* acl_src0, * acl_src1, * acl_dst;
    if (!ggml_are_same_shape(src0, src1) && ggml_ascend_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1);
        acl_src0 = ggml_ascend_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_ascend_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst = ggml_ascend_create_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = ggml_ascend_create_tensor(src0);
        acl_src1 = ggml_ascend_create_tensor(src1);
        acl_dst = ggml_ascend_create_tensor(dst);
    }

    ggml_ascend_add_fn(ctx, acl_src0, acl_src1, acl_dst);

    aclDestroyTensor(acl_src0);
    aclDestroyTensor(acl_src1);
    aclDestroyTensor(acl_dst);
}

void ggml_ascend_add_fn(ggml_backend_ascend_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    aclScalar* alpha(nullptr);
    float alphaValue(1.f);
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    auto workSpaceSize(uint64_t(0));
    aclOpExecutor* exe(nullptr);
    void* workSpaceAddr(nullptr);

    aclnnAddGetWorkspaceSize(acl_src0, acl_src1, alpha, acl_dst, &workSpaceSize, &exe);
    if (workSpaceSize > 0) {
        aclrtMalloc(&workSpaceAddr, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnAdd(workSpaceAddr, workSpaceSize, exe, ctx.stream());

    aclDestroyScalar(alpha);
    aclrtFree(workSpaceAddr);
} 

void ggml_ascend_mul(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    aclnn_shape_t selfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t otherShape = {ne13, ne12, ne11, ne10};
    aclnn_shape_t outShape = {
        (selfShape[0] > otherShape[0]) ? selfShape[0] : otherShape[0],
        (selfShape[1] > otherShape[1]) ? selfShape[1] : otherShape[1],
        (selfShape[2] > otherShape[2]) ? selfShape[2] : otherShape[2],
        (selfShape[3] > otherShape[3]) ? selfShape[3] : otherShape[3]
    };

    int ret = aclnn_mul_func(src0->data, src1->data, dst->data,
                            selfShape, otherShape, outShape,
                            ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul failed. ERROR: %d\n", ret); return);
}

void ggml_ascend_mul_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    auto src0(dst->src[0]), src1(dst->src[1]);
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    aclTensor* acl_src0, * acl_src1, *acl_dst;
    if (!ggml_are_same_shape(src0, src1) && ggml_ascend_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_ascend_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_ascend_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst = ggml_ascend_create_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = ggml_ascend_create_tensor(src0);
        acl_src1 = ggml_ascend_create_tensor(src1);
        acl_dst = ggml_ascend_create_tensor(dst);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnMulGetWorkspaceSize(acl_src0, acl_src1, acl_dst, &workspaceSize, &executor);

    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    auto stream(ctx.stream());
    aclnnMul(workspaceAddr, workspaceSize, executor, stream);

    aclDestroyTensor(acl_src0);
    aclDestroyTensor(acl_src1);
    aclDestroyTensor(acl_dst);
    aclrtFree(workspaceAddr);
}

void ggml_ascend_cpy(ggml_backend_ascend_context &ctx, ggml_tensor *src, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    aclnn_shape_t srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    aclnn_shape_t dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    srcShape = dstShape;

    // std::cout<<"in ggml_ascend_cpy :"<<std::endl;
    // std::cout<<"src shape: "<<src->ne[3]<<" "<<src->ne[2]<<" "<<src->ne[1]<<" "<<src->ne[0]<<std::endl;
    // std::cout<<"dst shape: "<<dst->ne[3]<<" "<<dst->ne[2]<<" "<<dst->ne[1]<<" "<<dst->ne[0]<<std::endl;
    // std::cout<<"src name: "<<src->name<<std::endl;
    // std::cout<<"dst name: "<<dst->name<<std::endl;


    // int ret = aclnn_cpy_func(dst->data, src->data,
    //                         dstShape, srcShape,
    //                         ggml_to_acl_map[dst->type], ggml_to_acl_map[src->type],
    //                         stream);

    int ret = aclnn_cpy_func(dst->data, src->data,
                            dstShape, srcShape,
                            ggml_to_acl_map[dst->type], ggml_to_acl_map[src->type],
                            stream,
                            dst->nb, src->nb);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_cpy failed. ERROR: %d\n", ret); return);
}

// void ggml_ascend_cpy_new(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
//     ggml_ascend_dup_new(ctx, dst);
// }

void ggml_ascend_dup(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor *src = dst->src[0];
    ggml_ascend_cpy(ctx, src, dst);
}

static void cann_copy(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnInplaceCopyGetWorkspaceSize(acl_dst, acl_src, &workspaceSize,
                                        &executor);

    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, ctx.stream());
    aclrtFree(workspaceAddr);
}

// void ggml_ascend_dup_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
//     auto src(dst->src[0]);
//     auto acl_src(ggml_ascend_create_tensor(src)), acl_dst(ggml_ascend_create_tensor(dst));

//     void *src_ptr, *dst_ptr;
//     aclrtMalloc(&src_ptr, sizeof(ggml_tensor), ACL_MEM_MALLOC_NORMAL_ONLY);
//     aclrtMalloc(&dst_ptr, sizeof(ggml_tensor), ACL_MEM_MALLOC_NORMAL_ONLY);

//     src->extra = src_ptr; dst->extra = dst_ptr;
//     aclrtMemcpy(src->extra, sizeof(ggml_tensor), src, sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE);
//     aclrtMemcpy(dst->extra, sizeof(ggml_tensor), dst, sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE);

//     if ((dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32) &&
//         ggml_are_same_shape(src, dst)) {
//         cann_copy(ctx, acl_src, acl_dst);
//         aclDestroyTensor(acl_src);
//         aclDestroyTensor(acl_dst);
//         return;
//     }

//     if (src->type == GGML_TYPE_F16) {
//         if (dst->type == GGML_TYPE_Q8_0) {
//             exit(1);
//         }
//         if (dst->type == GGML_TYPE_F16) {
//             if (ggml_are_same_shape(src, dst)) {
//                 cann_copy(ctx, acl_src, acl_dst);
//                 aclDestroyTensor(acl_src);
//                 aclDestroyTensor(acl_dst);
//                 return;
//             }
//             if (ggml_is_contiguous(dst)) {
//                 const size_t src_type_size = ggml_type_size(src->type);
//                 if (src->nb[0] == src_type_size) {
//                     // src0 is contigous on first dimension, copy by rows
//                     int64_t rows_num = ggml_nrows(src);

//                     aclrtlaunch_ascendc_dup_by_rows_fp16(
//                         rows_num, ctx.stream(), src->data, dst->data,
//                         ((ggml_tensor*)src->extra)->ne,
//                         ((ggml_tensor*)src->extra)->nb,
//                         ((ggml_tensor*)dst->extra)->ne,
//                         ((ggml_tensor*)dst->extra)->nb);
//                     return;
//                 }
//                 exit(2);
//             }
//             exit(2);
//         }
//         if (dst->type == GGML_TYPE_F32) {
//             if (ggml_are_same_shape(src, dst)) {
//                 cann_copy(ctx, acl_src, acl_dst);
//                 aclDestroyTensor(acl_src);
//                 aclDestroyTensor(acl_dst);
//                 return;
//             }
//             if (ggml_is_contiguous(dst)) {
//                 const size_t src_type_size = ggml_type_size(src->type);
//                 if (src->nb[0] == src_type_size) {
//                     // src0 is contigous on first dimension, copy by rows
//                     int64_t rows_num = ggml_nrows(src);
//                     aclrtlaunch_ascendc_dup_by_rows_fp16_to_fp32(
//                         rows_num, ctx.stream(), src->data, dst->data,
//                         ((ggml_tensor*)src->extra)->ne,
//                         ((ggml_tensor*)src->extra)->nb,
//                         ((ggml_tensor*)dst->extra)->ne,
//                         ((ggml_tensor*)dst->extra)->nb);
//                     return;
//                 }
//                 exit(2);
//             }
//             exit(2);
//         }
//         // TODO
//         exit(2);
//     } else if (src->type == GGML_TYPE_F32) {
//         // TODO: if (src0->type == dst->type && ne00 == ne0 && nb00 == type_size
//         //          && nb0 == type_size)
//         if (dst->type == GGML_TYPE_Q8_0) {
//             exit(1);
//         }
//         if (dst->type == GGML_TYPE_F32) {
//             if (ggml_are_same_shape(src, dst)) {
//                 cann_copy(ctx, acl_src, acl_dst);
//                 aclDestroyTensor(acl_src);
//                 aclDestroyTensor(acl_dst);
//                 return;
//             }
//             if (ggml_is_contiguous(dst)) {
//                 const size_t src_type_size = ggml_type_size(src->type);
//                 if (src->nb[0] == src_type_size) {
//                     // src0 is contigous on first dimension, copy by rows
//                     int64_t rows_num = ggml_nrows(src);
//                     aclrtlaunch_ascendc_dup_by_rows_fp32(
//                         rows_num, ctx.stream(), src->data, dst->data,
//                         ((ggml_tensor*)src->extra)->ne,
//                         ((ggml_tensor*)src->extra)->nb,
//                         ((ggml_tensor*)dst->extra)->ne,
//                         ((ggml_tensor*)dst->extra)->nb);
//                     return;
//                 }
//                 exit(2);
//             } else {
//                 // TODO: dst not contiguous
//                 exit(2);
//             }
//         }
//         if (dst->type == GGML_TYPE_F16) {
//             if (ggml_are_same_shape(src, dst)) {
//                 cann_copy(ctx, acl_src, acl_dst);
//                 aclDestroyTensor(acl_src);
//                 aclDestroyTensor(acl_dst);
//                 return;
//             }
//             if (ggml_is_contiguous(dst)) {
//                 const size_t src_type_size = ggml_type_size(src->type);
//                 if (src->nb[0] == src_type_size) {
//                     // src0 is contigous on first dimension, copy by rows
//                     int64_t rows_num = ggml_nrows(src);
//                     aclrtlaunch_ascendc_dup_by_rows_fp32_to_fp16(
//                         rows_num, ctx.stream(), src->data, dst->data,
//                         ((ggml_tensor*)src->extra)->ne,
//                         ((ggml_tensor*)src->extra)->nb,
//                         ((ggml_tensor*)dst->extra)->ne,
//                         ((ggml_tensor*)dst->extra)->nb);
//                     return;
//                 }
//                 exit(2);
//             }
//         }
//         // TODO
//         exit(2);
//     } else {
//         if (ggml_are_same_shape(src, dst)) {
//             cann_copy(ctx, acl_src, acl_dst);
//             aclDestroyTensor(acl_src);
//             aclDestroyTensor(acl_dst);
//             return;
//         }
//         exit(2);
//     }
// }

void ggml_ascend_silu(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src = dst->src[0];
    aclrtStream stream = ctx.stream();

    aclnn_shape_t srcShape = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
    aclnn_shape_t dstShape = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

    int ret = aclnn_silu_func(src->data, dst->data,
                            srcShape, dstShape,
                            ggml_to_acl_map[src->type], ggml_to_acl_map[dst->type],
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_silu failed. ERROR: %d\n", ret); return);
}

void ggml_ascend_silu_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    auto src(dst->src[0]);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = ggml_ascend_create_tensor(src);
    aclTensor* acl_dst = ggml_ascend_create_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnSiluGetWorkspaceSize(acl_src, acl_dst, &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    auto stream(ctx.stream());
    aclnnSilu(workspaceAddr, workspaceSize, executor, stream);

    aclDestroyTensor(acl_src);
    aclDestroyTensor(acl_dst);
    aclrtFree(workspaceAddr);
}

void ggml_ascend_rms_norm(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    aclrtStream stream = ctx.stream();

    aclnn_shape_t xShape = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    aclnn_shape_t gammaShape = {src0->ne[0]};
    aclnn_shape_t yShape = xShape;
    aclnn_shape_t rstdShape = {src0->ne[0], 1};

    void* gammaDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    std::vector<float> gammaHostData(src0->ne[0], 1);
    std::vector<float> rstdHostData = {1, float(src0->ne[0])};

    int ret = data_addr_malloc(gammaShape, gammaHostData, &gammaDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rms_norm failed. ERROR: %d\n", ret); return);
    ret = data_addr_malloc(rstdShape, rstdHostData, &rstdDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rms_norm failed. ERROR: %d\n", ret); return);

    float eps;
    memcpy(&eps, (float*)dst->op_params+0, sizeof(float));

    ret = aclnn_rms_norm_func(src0->data, gammaDeviceAddr, dst->data, rstdDeviceAddr,
                                xShape, gammaShape, yShape, rstdShape,
                                ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                eps, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rms_norm failed. ERROR: %d\n", ret); return);

    aclrtFree(gammaDeviceAddr);
    aclrtFree(rstdDeviceAddr);
}

void ggml_ascend_rms_norm_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    auto src(dst->src[0]);

    aclTensor* acl_src = ggml_ascend_create_tensor(src);
    aclTensor* acl_dst = ggml_ascend_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps > 0.0f);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    size_t one_tensor_n_bytes = src->ne[0] * ggml_element_size(src);
    // ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);
    void *one_ptr;
    aclrtMalloc(&one_ptr, one_tensor_n_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    aclTensor* acl_gamma = aclnn_ones(
        ctx, one_ptr, one_tensor_n_bytes, src->ne, 1,
        ggml_to_acl_map[src->type], ggml_element_size(src));

    size_t zero_tensor_n_bytes =
        src->ne[1] * src->ne[2] * src->ne[3] * ggml_element_size(src);
    // ggml_cann_pool_alloc zero_tensor_allocator(ctx.pool(), zero_tensor_n_bytes);
    void *zero_ptr;
    aclrtMalloc(&zero_ptr, zero_tensor_n_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor* acl_rstd =
        aclnn_zero(ctx, zero_ptr, zero_tensor_n_bytes,
                   src->ne, GGML_MAX_DIMS, ggml_to_acl_map[src->type],
                   ggml_element_size(src));

    aclnnRmsNormGetWorkspaceSize(
        acl_src, acl_gamma, eps, acl_dst, acl_rstd, &workspaceSize, &executor);

    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnRmsNorm(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclDestroyTensor(acl_src);
    aclDestroyTensor(acl_dst);
    aclDestroyTensor(acl_gamma);
    aclDestroyTensor(acl_rstd);
    aclrtFree(one_ptr);
    aclrtFree(zero_ptr);
    aclrtFree(workspaceAddr);
}

void ggml_ascend_soft_max_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    auto src0 = dst->src[0], src1 = dst->src[1];
    auto stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    auto dataAddr = src0->data;
    auto maskAddr = src1->data;
    auto outAddr = dst->data;

    float scale;
    memcpy((void *)&scale, (void *)&dst->op_params[0], sizeof(float));

    aclnn_shape_t dataShape{ne03, ne02, ne01, ne00};
    aclnn_shape_t maskShape{ne12, ne01, ne10};
    aclnn_shape_t outShape{ne3, ne2, ne1, ne0};

    auto ret = aclnn_soft_max_func(dataAddr, maskAddr, scale, outAddr, dataShape, maskShape, outShape, 
    ggml_to_acl_map[src0->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type], stream);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s failed. ERROR: %d\n", __func__, ret); return);
}

void ggml_ascend_soft_max(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0]; // 
    ggml_tensor* src1 = dst->src[1]; // mask
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    void* mulsSelfDeviceAddr = src0->data;
    void* mulsOutDeviceAddr = nullptr;
    // float scale = dst->op_params[0];
    float scale;
    memcpy((void *)&scale, (void *)&dst->op_params[0], sizeof(float));
    aclnn_shape_t mulsSelfShape = {ne00, ne01, ne02, ne03};
    aclnn_shape_t mulsOutShape = mulsSelfShape;
    std::vector<float> mulsOutHostData(ne00 * ne01 * ne02 * ne03, 0);
    auto ret = data_addr_malloc(mulsOutShape, mulsOutHostData, &mulsOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    ret = aclnn_muls_func(mulsSelfDeviceAddr, mulsOutDeviceAddr, mulsSelfShape, mulsOutShape, ACL_FLOAT, ACL_FLOAT, scale, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\nmulsOut: \n");
    // auto tmp_size = GetShapeSize(mulsOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), mulsOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* addSelfDeviceAddr = mulsOutDeviceAddr;
    void* addOtherDeviceAddr = src1->data;
    void* addOutDeviceAddr = nullptr;
    aclnn_shape_t addSelfShape = mulsOutShape;
    aclnn_shape_t addOtherShape = {ne10, ne01, ne12, ne13};
    aclnn_shape_t addOutShape = {
        (addSelfShape[0] > addOtherShape[0]) ? addSelfShape[0] : addOtherShape[0],
        (addSelfShape[1] > addOtherShape[1]) ? addSelfShape[1] : addOtherShape[1],
        (addSelfShape[2] > addOtherShape[2]) ? addSelfShape[2] : addOtherShape[2],
        (addSelfShape[3] > addOtherShape[3]) ? addSelfShape[3] : addOtherShape[3]
    };
    std::vector<float> addOutHostData(addOutShape[0] * addOutShape[1] * addOutShape[2] * addOutShape[3], 0);
    ret = data_addr_malloc(addOutShape, addOutHostData, &addOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    ret = aclnn_add_func(addSelfDeviceAddr, addOtherDeviceAddr, addOutDeviceAddr,
                        addSelfShape, addOtherShape, addOutShape,
                        ACL_FLOAT, ggml_to_acl_map[src1->type], ACL_FLOAT,
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\naddOut: \n");
    // auto tmp_size = GetShapeSize(addOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), addOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* softMaxSelfDeviceAddr = addOutDeviceAddr;
    void* softMaxOutDeviceAddr = dst->data;
    aclnn_shape_t softMaxSelfShape = addOutShape;
    aclnn_shape_t softMaxOutShape = softMaxSelfShape;
    ret = aclnn_soft_max_func(softMaxSelfDeviceAddr, softMaxOutDeviceAddr,
                            softMaxSelfShape, softMaxOutShape,
                            ACL_FLOAT, ACL_FLOAT,
                            stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_soft_max failed. ERROR: %d\n", ret); return);

    aclrtFree(mulsOutDeviceAddr);
    aclrtFree(addOutDeviceAddr);
}

static void aclnn_softmax(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                          int64_t dim, aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnSoftmaxGetWorkspaceSize(acl_src, dim, acl_dst,
                                           &workspaceSize, &executor);

    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclrtStream stream = ctx.stream();
    aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream);

    aclrtFree(workspaceAddr);
}

static void aclnn_muls(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                       float scale, aclTensor* acl_dst, bool inplace) {
    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    if (inplace) {
        aclnnInplaceMulsGetWorkspaceSize(acl_src, acl_scale,
                                                   &workspaceSize, &executor);
        if (workspaceSize > 0) {
            aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnInplaceMuls(workspaceAddr, workspaceSize, executor,
                                   ctx.stream());
    } else {
        aclnnMulsGetWorkspaceSize(acl_src, acl_scale, acl_dst,
                                            &workspaceSize, &executor);
        if (workspaceSize > 0) {
            aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }


        aclnnMuls(workspaceAddr, workspaceSize, executor, ctx.stream());
    }

    aclDestroyScalar(acl_scale);
    aclrtFree(workspaceAddr);
}

static void aclnn_cast(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                       aclTensor* acl_dst, aclDataType cast_data_type) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnCastGetWorkspaceSize(acl_src, cast_data_type, acl_dst,
                                        &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnCast(workspaceAddr, workspaceSize, executor, ctx.stream());
    aclrtFree(workspaceAddr);
}

static void aclnn_pow_tensor_tensor(ggml_backend_ascend_context& ctx,
                                    aclTensor* acl_dst, aclTensor* acl_exp) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnInplacePowTensorTensorGetWorkspaceSize(
        acl_dst, acl_exp, &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize,
                                          executor, ctx.stream());
    aclrtFree(workspaceAddr);
}

static void aclnn_alibi(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                        aclTensor* acl_position, aclTensor* acl_dst,
                        const int n_head, int64_t* src_ne, const size_t src_nb0,
                        float max_bias, ggml_tensor* dst) {
    const int64_t ne2_ne3 = src_ne[2] * src_ne[3];
    GGML_ASSERT(src_nb0 == sizeof(float));
    GGML_ASSERT(n_head == src_ne[2]);

    const int n_heads_log2_floor = 1u << (uint32_t)floor(log2(n_head));

    float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

    // init arange
    void* tmp_arange_buffer(nullptr);
    aclrtMalloc(&tmp_arange_buffer, ne2_ne3 * ggml_type_size(dst->type), ACL_MEM_MALLOC_NORMAL_ONLY);

    // arange1: [1, ..., n_heads_log2_floor+1)
    float start = 1;
    float stop = n_heads_log2_floor + 1;
    float step = 1;
    int64_t n_elements_arange = n_heads_log2_floor;

    int64_t tmp_arange1_ne[] = {n_heads_log2_floor};
    size_t tmp_arange1_nb[] = {sizeof(dst->type)};
    aclTensor* tmp_arange1_tensor = ggml_ascend_create_tensor(
        tmp_arange_buffer, ggml_to_acl_map[dst->type],
        ggml_type_size(dst->type), tmp_arange1_ne, tmp_arange1_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange1_tensor, start, stop, step, n_elements_arange);

    aclTensor* tmp_arange2_tensor = nullptr;
    if (n_heads_log2_floor < ne2_ne3) {
        // arange2: [1, ..., 2 * (k - n_heads_log2_floor) + 1)
        start = 1;
        stop = 2 * (ne2_ne3 - n_heads_log2_floor) + 1;
        step = 2;
        n_elements_arange = ne2_ne3 - n_heads_log2_floor;
        int64_t tmp_arange2_ne[] = {ne2_ne3 - n_heads_log2_floor};
        size_t tmp_arange2_nb[] = {sizeof(dst->type)};

        aclTensor* tmp_arange2_tensor = ggml_ascend_create_tensor(
            (char*)tmp_arange_buffer +
                n_heads_log2_floor * ggml_type_size(dst->type),
            ggml_to_acl_map[dst->type], ggml_type_size(dst->type),
            tmp_arange2_ne, tmp_arange2_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
        aclnn_arange(ctx, tmp_arange2_tensor, start, stop, step,
                     n_elements_arange);
    }

    // init mk_base
    void* tmp_mk_base_buffer(nullptr);
    aclrtMalloc(&tmp_mk_base_buffer, ne2_ne3 * ggml_type_size(dst->type), ACL_MEM_MALLOC_NORMAL_ONLY);
    int64_t tmp_mk_base1_ne[] = {n_heads_log2_floor};
    size_t tmp_mk_base1_nb[] = {sizeof(dst->type)};
    aclTensor* tmp_mk_base1_tensor = ggml_ascend_create_tensor(
        tmp_mk_base_buffer, ggml_to_acl_map[dst->type],
        ggml_type_size(dst->type), tmp_mk_base1_ne, tmp_mk_base1_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_fill_scalar(ctx, m0, tmp_mk_base1_tensor);

    aclTensor* tmp_mk_base2_tensor = nullptr;
    if (n_heads_log2_floor < ne2_ne3) {
        int64_t tmp_mk_base2_ne[] = {ne2_ne3 - n_heads_log2_floor};
        size_t tmp_mk_base2_nb[] = {sizeof(dst->type)};
        aclTensor* tmp_mk_base2_tensor = ggml_ascend_create_tensor(
            (char*)tmp_mk_base_buffer +
                n_heads_log2_floor * ggml_type_size(dst->type),
            ggml_to_acl_map[dst->type], ggml_type_size(dst->type),
            tmp_mk_base2_ne, tmp_mk_base2_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
        aclnn_fill_scalar(ctx, m1, tmp_mk_base2_tensor);
    }

    // init mk
    int64_t tmp_mk_base_ne[] = {ne2_ne3};
    size_t tmp_mk_base_nb[] = {sizeof(dst->type)};
    aclTensor* tmp_mk_base_tensor = ggml_ascend_create_tensor(
        tmp_mk_base_buffer, ggml_to_acl_map[dst->type],
        ggml_type_size(dst->type), tmp_mk_base_ne, tmp_mk_base_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
    aclTensor* tmp_arange_tensor = ggml_ascend_create_tensor(
        tmp_arange_buffer, ggml_to_acl_map[dst->type],
        ggml_type_size(dst->type), tmp_mk_base_ne, tmp_mk_base_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
    aclnn_pow_tensor_tensor(ctx, tmp_mk_base_tensor, tmp_arange_tensor);

    // reshape mk
    int64_t tmp_mk_ne[] = {1, 1, src_ne[2], src_ne[3]};
    size_t tmp_mk_nb[GGML_MAX_DIMS];
    tmp_mk_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mk_nb[i] = tmp_mk_nb[i - 1] * tmp_mk_ne[i - 1];
    }
    aclTensor* tmp_mk_tensor = ggml_ascend_create_tensor(
        tmp_mk_base_buffer, ggml_to_acl_map[dst->type],
        ggml_type_size(dst->type), tmp_mk_ne, tmp_mk_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);

    // acl_position * mk
    int64_t tmp_output_ne[] = {src_ne[0], src_ne[1], src_ne[2], src_ne[3]};
    size_t tmp_output_nb[GGML_MAX_DIMS];
    tmp_output_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_output_nb[i] = tmp_output_nb[i - 1] * tmp_output_ne[i - 1];
    }
    // ggml_cann_pool_alloc output_allocator(ctx.pool(), ggml_nbytes(dst));
    void* tmp_output_buffer(nullptr);
    aclrtMalloc(&tmp_output_buffer, ggml_nbytes(dst), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclTensor* tmp_output_tensor = ggml_ascend_create_tensor(
        tmp_output_buffer, ggml_to_acl_map[dst->type],
        ggml_type_size(dst->type), tmp_output_ne, tmp_output_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);
    ggml_ascend_add_fn(ctx, acl_position, tmp_mk_tensor, tmp_output_tensor);

    // add
    ggml_ascend_add_fn(ctx, tmp_output_tensor, acl_src, acl_dst);

    aclDestroyTensor(tmp_arange1_tensor);
    aclDestroyTensor(tmp_arange2_tensor);
    aclDestroyTensor(tmp_mk_base1_tensor);
    aclDestroyTensor(tmp_mk_base2_tensor);
    aclDestroyTensor(tmp_mk_base_tensor);
    aclDestroyTensor(tmp_arange_tensor);
    aclDestroyTensor(tmp_mk_tensor);
    aclDestroyTensor(tmp_output_tensor);

    aclrtFree(tmp_arange_buffer);
    aclrtFree(tmp_mk_base_buffer);
    aclrtFree(tmp_output_buffer);
}

void ggml_ascend_softmax_renew(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];  // mask

    aclTensor* acl_src0 = ggml_ascend_create_tensor(src0);
    aclTensor* acl_dst = ggml_ascend_create_tensor(dst);

    float scale = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float*)dst->op_params + 1, sizeof(float));

    // input mul scale
    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);

    size_t n_bytes = ggml_nbytes(src0);
    // ggml_cann_pool_alloc mul_scale_allocator(ctx.pool(), n_bytes);

    void *scale_ptr;
    aclrtMalloc(&scale_ptr, n_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    // void* input_mul_scale_buffer = mul_scale_allocator.get();
    aclTensor* acl_input_mul_scale_tensor = ggml_ascend_create_tensor(
        scale_ptr, ACL_FLOAT, ggml_type_size(src0->type), src0->ne,
        src0->nb, GGML_MAX_DIMS);

    bool inplace = false;
    aclnn_muls(ctx, acl_src0, scale, acl_input_mul_scale_tensor, inplace);

    // mask
    aclTensor* acl_src1_fp32_tensor = nullptr;
    aclTensor* tmp_mask_tensor = nullptr;
    // ggml_cann_pool_alloc src1_fp32_allocator(ctx.pool());
    if (src1) {
        const bool use_f16 = src1->type == GGML_TYPE_F16;
        if (use_f16) {
            // cast to fp32
            size_t n_bytes = ggml_nelements(src1) * sizeof(float_t);
            size_t src1_fp32_nb[GGML_MAX_DIMS];
            src1_fp32_nb[0] = sizeof(float_t);
            for (int i = 1; i < GGML_MAX_DIMS; i++) {
                src1_fp32_nb[i] = src1_fp32_nb[i - 1] * src1->ne[i - 1];
            }
            void* src1_fp32_buffer(nullptr);
            aclrtMalloc(&src1_fp32_buffer, n_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
            acl_src1_fp32_tensor = ggml_ascend_create_tensor(
                src1_fp32_buffer, ACL_FLOAT, sizeof(float), src1->ne,
                src1_fp32_nb, GGML_MAX_DIMS);
            aclTensor* acl_src1 = ggml_ascend_create_tensor(src1);
            aclnn_cast(ctx, acl_src1, acl_src1_fp32_tensor, ACL_FLOAT);

            aclDestroyTensor(acl_src1);
        } else {
            acl_src1_fp32_tensor = ggml_ascend_create_tensor(src1);
        }

        // broadcast the mask across rows, only use ne11 of ne01 in mask
        if (src1->ne[1] != src0->ne[1]) {
            // mask shape: [1,1,ne11,ne10]
            int64_t tmp_mask_ne[] = {src0->ne[0], src0->ne[1], 1, 1};
            size_t tmp_mask_nb[GGML_MAX_DIMS];
            tmp_mask_nb[0] = sizeof(float_t);
            for (int i = 1; i < GGML_MAX_DIMS; i++) {
                tmp_mask_nb[i] = tmp_mask_nb[i - 1] * tmp_mask_ne[i - 1];
            }
            tmp_mask_tensor = ggml_ascend_create_tensor(
                src1->data, ACL_FLOAT, sizeof(float), tmp_mask_ne, tmp_mask_nb,
                GGML_MAX_DIMS, ACL_FORMAT_ND);
        }

        // alibi
        const int n_head = src0->ne[2];
        const size_t src_nb0 = src0->nb[0];

        n_bytes = ggml_nbytes(dst);
        // ggml_cann_pool_alloc output_allocator(ctx.pool(), n_bytes);
        void* output_buffer(nullptr);
        aclrtMalloc(&output_buffer, n_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclTensor* alibi_output_tensor = ggml_ascend_create_tensor(
            output_buffer, ACL_FLOAT, ggml_type_size(dst->type), dst->ne,
            dst->nb, GGML_MAX_DIMS);
        if (max_bias <= 0.0f) {
            // slope = 1.0
            if (tmp_mask_tensor) {
                ggml_ascend_add_fn(ctx, tmp_mask_tensor, acl_input_mul_scale_tensor,
                          alibi_output_tensor);
            } else {
                ggml_ascend_add_fn(ctx, acl_src1_fp32_tensor, acl_input_mul_scale_tensor,
                          alibi_output_tensor);
            }
        } else {
            // slope != 1.0
            if (tmp_mask_tensor) {
                aclnn_alibi(ctx, acl_input_mul_scale_tensor, tmp_mask_tensor,
                            alibi_output_tensor, n_head, src0->ne, src_nb0,
                            max_bias, dst);
            } else {
                aclnn_alibi(ctx, acl_input_mul_scale_tensor,
                            acl_src1_fp32_tensor, alibi_output_tensor, n_head,
                            src0->ne, src_nb0, max_bias, dst);
            }
        }

        // softmax
        aclnn_softmax(ctx, alibi_output_tensor, 3, acl_dst);
        aclDestroyTensor(alibi_output_tensor);
        aclrtFree(output_buffer);
    } else {
        aclnn_softmax(ctx, acl_input_mul_scale_tensor, 3, acl_dst);
    }

    aclDestroyTensor(acl_src0);
    aclDestroyTensor(acl_src1_fp32_tensor);
    aclDestroyTensor(acl_dst);
    aclDestroyScalar(acl_scale);
    aclDestroyTensor(acl_input_mul_scale_tensor);
    aclDestroyTensor(tmp_mask_tensor);

    aclrtFree(scale_ptr);
}

void ggml_ascend_mul_mat(ggml_backend_ascend_context &ctx, ggml_tensor *src0, ggml_tensor *src1, ggml_tensor *dst) {
    aclrtStream stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    // std::cout<<"in ggml_ascend_mat_mul :"<<std::endl;
    // std::cout<<"src0 shape: "<<src0->ne[3]<<" "<<src0->ne[2]<<" "<<src0->ne[1]<<" "<<src0->ne[0]<<std::endl;
    // std::cout<<"src1 shape: "<<src1->ne[3]<<" "<<src1->ne[2]<<" "<<src1->ne[1]<<" "<<src1->ne[0]<<std::endl;
    // std::cout<<"dst shape: "<<dst->ne[3]<<" "<<dst->ne[2]<<" "<<dst->ne[1]<<" "<<dst->ne[0]<<std::endl;
    // std::cout<<"src name: "<<src0->name<<" "<<src1->name<<std::endl;
    // std::cout<<"dst name: "<<dst->name<<std::endl;
    // std::cout<<"src0 type: "<< src0->type << std::endl;
    // std::cout<<"src1 type: "<< src1->type << std::endl;
    // std::cout<<"dst type: "<< dst->type << std::endl;


    CHECK_RET((ne03 == ne13 && ne13 == 1), LOG_PRINT("error: ne03: %ld, ne13: %ld\n", ne03, ne13));
    CHECK_RET((ne12 >= ne02 && ne12 % ne02 == 0), LOG_PRINT("error: ne12: %ld, ne02: %ld\n", ne12, ne02));

    void* permuteSelfDeviceAddr = src0->data;
    void* permuteOutDeviceAddr = nullptr;
    aclnn_shape_t permuteSelfShape = {ne03, ne02, ne01, ne00};
    aclnn_shape_t permuteOutShape = {ne03, ne02, ne00, ne01};
    aclnn_shape_t permuteDims = {0, 1, 3, 2};
    int ret = addr_malloc(permuteOutShape, &permuteOutDeviceAddr, ggml_type_size_t[src0->type]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    ret = aclnn_permute_func(permuteSelfDeviceAddr, permuteOutDeviceAddr,
                                permuteSelfShape, permuteOutShape,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src0->type],
                                permuteDims, stream);

    // LOG_PRINT("\naclnn_permute_func: \n");
    // auto tmp_size = GetShapeSize(permuteOutShape);
    // std::vector<aclFloat16> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), permuteOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    // }

    void* cpySelfRefDeviceAddr = nullptr;
    void* cpySrcDeviceAddr = permuteOutDeviceAddr;
    aclnn_shape_t cpySelfRefShape = {ne03, ne02, ne00, ne01};
    aclnn_shape_t cpySrcShape = permuteOutShape;
    ret = addr_malloc(cpySelfRefShape, &cpySelfRefDeviceAddr, ggml_type_size_t[src1->type]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    ret = aclnn_cpy_func(cpySelfRefDeviceAddr, cpySrcDeviceAddr,
                        cpySelfRefShape, cpySrcShape,
                        ggml_to_acl_map[src1->type], ggml_to_acl_map[src0->type],
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    void* boardcastSrc0DeviceAddr = nullptr;
    int64_t size = ne03 * ne12 * ne01 * ne00;
    ret = aclrtMalloc(&boardcastSrc0DeviceAddr, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);
    for (int i = 0; i < size; i += ne02 * ne01 * ne00) {
        ret = aclrtMemcpy((void *)((float *)boardcastSrc0DeviceAddr + i), ne02 * ne01 * ne00 * sizeof(float), cpySelfRefDeviceAddr, ne02 * ne01 * ne00 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    }

    // LOG_PRINT("\boardcastSrc0DeviceAddr: \n");
    // auto tmp_size = size;
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), boardcastSrc0DeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    // LOG_PRINT("\aclnn_cpy_func: \n");
    // auto tmp_size = GetShapeSize(cpySelfRefShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cpySelfRefDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* batchMatMulSelfDeviceAddr = src1->data;
    void* batchMatMulMat2DeviceAddr = boardcastSrc0DeviceAddr;
    void* batchMatMulOutDeviceAddr = dst->data;
    aclnn_shape_t batchMatMulSelfShape = {ne12, ne11, ne10};
    aclnn_shape_t batchMatMulMat2Shape = {ne12, ne00, ne01};
    aclnn_shape_t batchMatMulOutShape = {ne2, ne1, ne0};

    // LOG_PRINT("ne12: %ld, ne11: %ld, ne10: %ld\n", ne12, ne11, ne10);
    // LOG_PRINT("ne02: %ld, ne00: %ld, ne01: %ld\n", ne02, ne00, ne01);
    // LOG_PRINT("ne2: %ld, ne1: %ld, ne0: %ld\n", ne2, ne1, ne0);

    // LOG_PRINT("\abatchMatMulSelfDeviceAddr: \n");
    // auto tmp_size = GetShapeSize(batchMatMulSelfShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), batchMatMulSelfDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    ret = aclnn_batch_mat_mul_func(batchMatMulSelfDeviceAddr, batchMatMulMat2DeviceAddr, batchMatMulOutDeviceAddr,
                                batchMatMulSelfShape, batchMatMulMat2Shape, batchMatMulOutShape,
                                ggml_to_acl_map[src1->type], ggml_to_acl_map[src1->type], ggml_to_acl_map[dst->type],
                                stream);

    aclrtFree(permuteOutDeviceAddr);
    aclrtFree(cpySelfRefDeviceAddr);
    aclrtFree(boardcastSrc0DeviceAddr);
}

static void aclnn_mat_mul(ggml_backend_ascend_context& ctx, aclTensor* acl_input,
                          aclTensor* acl_weight, aclTensor* acl_dst) {
    int8_t cube_math_type = 1;  // ALLOW_FP32_DOWN_PRECISION, when input is
                                // fp32, atlas a2 will transpose it to HFLOAT32.

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnMatmulGetWorkspaceSize(acl_input, acl_weight, acl_dst,
                                          cube_math_type, &workspaceSize,
                                          &executor);

    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnMatmul(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclrtFree(workspaceAddr);
}


static void ggml_ascend_mul_mat_fn(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    ggml_tensor* weight = dst->src[0];  // weight
    ggml_tensor* input = dst->src[1];   // input

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(input, weight, dst);

    // transpose weight: [1,2,3,4] -> [1,2,4,3]
    int64_t transpose_ne[] = {bcast_weight_ne[1], bcast_weight_ne[0],
                              bcast_weight_ne[2], bcast_weight_ne[3],
                              bcast_weight_ne[4], bcast_weight_ne[5]};
    size_t transpose_nb[] = {bcast_weight_nb[1], bcast_weight_nb[0],
                             bcast_weight_nb[2], bcast_weight_nb[3],
                             bcast_weight_nb[4], bcast_weight_nb[5]};

    aclTensor* acl_weight_tensor =
        ggml_ascend_create_tensor(weight, transpose_ne, transpose_nb, bcast_dims);
    aclTensor* acl_input_tensor =
        ggml_ascend_create_tensor(input, BCAST_MUL_MAT_PARAM(input));
    aclTensor* acl_dst = ggml_ascend_create_tensor(dst, BCAST_MUL_MAT_PARAM(dst));
    aclnn_mat_mul(ctx, acl_input_tensor, acl_weight_tensor, acl_dst);

    aclDestroyTensor(acl_weight_tensor);
    aclDestroyTensor(acl_input_tensor);
    aclDestroyTensor(acl_dst);
}

void ggml_ascend_mul_mat_new(ggml_backend_ascend_context & ctx, ggml_tensor * dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type)
    {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
        ggml_ascend_mul_mat_fn(ctx, dst);
        break;
    default:
        break;
    }
}

void ggml_ascend_rope(ggml_backend_ascend_context &ctx, ggml_tensor *dst) {

    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclrtStream stream = ctx.stream();

    int64_t* ne = src0->ne;
    float freq_scale = dst->op_params[6];
    float freq_base = dst->op_params[5];
    int n_dims = dst->op_params[1];
    void* pos = src1->data;

    int64_t size = ne[0] * ne[1] * ne[2] * ne[3];


    // std::cout<<"in Rope"<<std::endl;
    // std::cout<<ne[0]<<" "<<ne[1]<<" "<<ne[2]<<" "<<ne[3]<<std::endl;

    // float theta_scale_pow[ne[0] / 2];
    // float theta_base[size];
    // float theta[size];
    // float sin_d[size];
    // float cos_d[size];

    float theta_scale = pow(freq_base, -2.0 / n_dims);
    // LOG_PRINT("theta_scale: %f\n", theta_scale);

    std::vector<float> powExpHostData(ne[0]/2, 0);
    std::vector<float> powOutData = powExpHostData;
    aclnn_shape_t powExpShape = {ne[0]/2, 1};
    aclnn_shape_t powOutShape = powExpShape;
    for(decltype(powExpHostData.size()) i(0); i < powExpHostData.size(); i++){
        powExpHostData[i] = i;
    }
    void* powExpDeviceAddr = nullptr;
    void* powOutDeviceAddr = nullptr;
    int ret = data_addr_malloc(powExpShape, powExpHostData, &powExpDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    ret = data_addr_malloc(powOutShape, powOutData, &powOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    ret = aclnn_pow_scalar_tensor_func(theta_scale, powExpDeviceAddr, powOutDeviceAddr,
        powExpShape, powOutShape,
        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\npowOut: \n");
    // auto tmp_size = GetShapeSize(powOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), powOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* mulOtherDeviceAddr = nullptr;
    ret = aclrtMalloc(&mulOtherDeviceAddr, size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // for (int i = 0; i < size; i += ne[0] / 2) {
    //     ret = aclrtMemcpy((void *)((float *)mulOtherDeviceAddr + i), ne[0] / 2 * sizeof(float), powOutDeviceAddr, ne[0] / 2 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // }
    aclnn_shape_t repeatSelfShape = {1, 1, ne[0]/2};
    aclnn_shape_t repeatOutShape = {ne[2], ne[1], ne[0]};
    std::vector<int64_t> repeatsArray = {ne[2], ne[1], 2};
    ret = aclnn_repeat_func(powOutDeviceAddr, mulOtherDeviceAddr,
                            repeatSelfShape, repeatOutShape,
                            ACL_FLOAT, ACL_FLOAT,
                            repeatsArray, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // std::vector<float> mulOtherHostData(size, 0);
    // std::generate_n(mulOtherHostData.begin(), size, [&, index = 0]() mutable {
    //     return theta_scale_pow[index++ % (ne[0]/2)];
    // });
    // for (int i=0; i< mulOtherHostData.size(); i++){
    //   std::cout << mulOtherHostData[i]<< " "; 
    // }
    // return;

    aclnn_shape_t mulSelfShape = {size, 1};
    aclnn_shape_t mulOtherShape = mulSelfShape;
    aclnn_shape_t mulOutShape = mulSelfShape;

    void* mulOutDeviceAddr = nullptr;
    void* mulSelfDeviceAddr = nullptr;
    std::vector<float> mulOutHostData(size, 0);
    ret = data_addr_malloc(mulOutShape, mulOutHostData, &mulOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    ret = aclrtMalloc(&mulSelfDeviceAddr, size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // for (int i = 0; i < size; i += ne[2]) {
    //     ret = aclrtMemcpy((void *)((int32_t *)mulSelfDeviceAddr + i), ne[2] * sizeof(int32_t), pos, ne[2] * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // }
    aclnn_shape_t repeatSelfShape2 = {ne[2], 1, 1};
    aclnn_shape_t repeatOutShape2 = {ne[2], ne[1], ne[0]};
    std::vector<int64_t> repeatsArray2 = {1, ne[1], ne[0]};
    ret = aclnn_repeat_func(pos, mulSelfDeviceAddr,
                            repeatSelfShape2, repeatOutShape2,
                            ACL_INT32, ACL_INT32,
                            repeatsArray2, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // std::vector<float> mulSelfHostData(size, 0);
    // std::generate_n(mulSelfHostData.begin(), size, [&, index = 0]() mutable {
    //     return (float)pos[index++ % ne[2]];
    // });
    // for (int i=0; i< mulSelfHostData.size(); i++){
    //   std::cout << mulSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_mul_func(mulSelfDeviceAddr, mulOtherDeviceAddr, mulOutDeviceAddr,
                        mulSelfShape, mulOtherShape, mulOutShape,
                        ACL_INT32, ACL_FLOAT, ACL_FLOAT,
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // LOG_PRINT("\nmulOut: \n");
    // auto tmp_size = GetShapeSize(mulOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), mulOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    // ret = aclnnMulFunc(mulSelfShape, mulOtherShape, mulOutShape, mulSelfHostData, mulOtherHostData, mulOutHostData, theta_base, context, stream);

    void* mulsOutDeviceAddr = nullptr;
    void* mulsSelfDeviceAddr = mulOutDeviceAddr;
    aclnn_shape_t mulsSelfShape = mulSelfShape;
    aclnn_shape_t mulsOutShape = mulOutShape;
    std::vector<float> mulsOutHostData(size, 0);
    ret = data_addr_malloc(mulsOutShape, mulsOutHostData, &mulsOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // std::vector<float> mulsSelfHostData(theta_base, theta_base+ size);

    // for (int i=0; i< mulsSelfHostData.size(); i++){
    //   std::cout << mulsSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_muls_func(mulsSelfDeviceAddr, mulsOutDeviceAddr, mulsSelfShape, mulsOutShape, ACL_FLOAT, ACL_FLOAT, freq_scale, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnMulsFunc(mulsSelfHostData, mulsOutHostData, freq_scale, mulsSelfShape, mulsOutShape, theta, context, stream);

    // LOG_PRINT("\nmulsOut: \n");
    // auto tmp_size = GetShapeSize(mulsOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), mulsOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* sinSelfDeviceAddr = mulsOutDeviceAddr;
    void* sinOutDeviceAddr = nullptr;
    aclnn_shape_t sinSelfShape = {size, 1};
    aclnn_shape_t sinOutShape = {size, 1};
    std::vector<float> sinOutHostData(size, 0);
    ret = data_addr_malloc(sinOutShape, sinOutHostData, &sinOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // std::vector<float> sinSelfHostData(theta, theta+ size);


    // for (int i=0; i< sinSelfHostData.size(); i++){
    //   std::cout << sinSelfHostData[i]<< " "; 
    // }
    // return;

    ret = aclnn_sin_func(sinSelfDeviceAddr, sinOutDeviceAddr, sinSelfShape, sinOutShape, ACL_FLOAT, ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnSinFunc(sinSelfShape, sinShape, sinSelfHostData, sinHostData, sin_d, context, stream);

    // LOG_PRINT("\nsinOut: \n");
    // auto tmp_size = GetShapeSize(sinOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), sinOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* cosSelfDeviceAddr = mulsOutDeviceAddr;
    void* cosOutDeviceAddr = nullptr;
    aclnn_shape_t cosSelfShape = {size, 1};
    aclnn_shape_t cosOutShape = {size, 1};
    std::vector<float> cosOutHostData(size, 0);
    ret = data_addr_malloc(cosOutShape, cosOutHostData, &cosOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // std::vector<float> cosSelfHostData(theta, theta+ size);

    ret = aclnn_cos_func(cosSelfDeviceAddr, cosOutDeviceAddr, cosSelfShape, cosOutShape, ACL_FLOAT, ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnCosFunc(cosSelfShape, cosShape, cosSelfHostData, cosHostData, cos_d, context, stream);

    // LOG_PRINT("\ncosOut: \n");
    // auto tmp_size = GetShapeSize(cosOutShape);
    // std::vector<float> resultData(tmp_size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cosOutDeviceAddr,
    //                     tmp_size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    // for (int64_t i = 0; i < tmp_size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }

    void* permuteSelfDeviceAddr = src0->data;
    void* permuteOutDeviceAddr = nullptr;
    aclnn_shape_t permuteSelfShape = {1, size/ne[0], ne[0]/2, 2};
    aclnn_shape_t permuteOutShape = {1, size/ne[0], 2, ne[0]/2};
    aclnn_shape_t permuteDims = {0, 1, 3, 2};
    ret = addr_malloc(permuteOutShape, &permuteOutDeviceAddr, ggml_type_size_t[src0->type]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_mul_mat failed. ERROR: %d\n", ret); return);

    ret = aclnn_permute_func(permuteSelfDeviceAddr, permuteOutDeviceAddr,
                                permuteSelfShape, permuteOutShape,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src0->type],
                                permuteDims, stream);

    void* queryDeviceAddr = permuteOutDeviceAddr;
    void* keyDeviceAddr = dst->data;
    void* sinDeviceAddr = sinOutDeviceAddr;
    void* cosDeviceAddr = cosOutDeviceAddr;
    aclnn_shape_t queryShape = {1, size/ne[0], 1, ne[0]};
    aclnn_shape_t keyShape = queryShape;
    aclnn_shape_t sinShapeRp = {1, size/ne[0], 1, ne[0]};
    aclnn_shape_t cosShapeRp = sinShapeRp;

    ret = aclnn_rope_func(queryDeviceAddr, keyDeviceAddr, cosDeviceAddr, sinDeviceAddr,
                        queryShape, keyShape, cosShapeRp, sinShapeRp,
                        ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                        stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);

    // ret = aclrtMemcpy(keyDeviceAddr, size * sizeof(float), queryDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ggml_ascend_rope failed. ERROR: %d\n", ret); return);
    // ret = aclnnRoPEFunc(queryShape, keyShape, cosShapeRp, sinShapeRp, queryHostData, keyHostData, cosQKHostData, sinQKHostData, dst, context, stream); 

    void* permuteSelfDeviceAddr1 = queryDeviceAddr;
    void* permuteOutDeviceAddr1 = keyDeviceAddr;
    aclnn_shape_t permuteSelfShape1 = {1, size/ne[0], 2, ne[0]/2};
    aclnn_shape_t permuteOutShape1 = {1, size/ne[0], ne[0]/2, 2};
    aclnn_shape_t permuteDims1 = {0, 1, 3, 2};

    ret = aclnn_permute_func(permuteSelfDeviceAddr1, permuteOutDeviceAddr1,
                                permuteSelfShape1, permuteOutShape1,
                                ggml_to_acl_map[src0->type], ggml_to_acl_map[src0->type],
                                permuteDims1, stream);

    aclrtFree(powExpDeviceAddr);
    aclrtFree(powOutDeviceAddr);
    aclrtFree(mulOtherDeviceAddr);
    aclrtFree(mulSelfDeviceAddr);
    aclrtFree(mulOutDeviceAddr);
    aclrtFree(mulsOutDeviceAddr);
    aclrtFree(sinOutDeviceAddr);
    aclrtFree(cosOutDeviceAddr);
    aclrtFree(permuteOutDeviceAddr);
}

static void aclnn_roll(ggml_backend_ascend_context& ctx, aclTensor* acl_src,
                       aclTensor* acl_dst, int64_t* shifts, int64_t* dims) {
    aclIntArray* acl_shifts = aclCreateIntArray(shifts, 1);
    aclIntArray* acl_dims = aclCreateIntArray(dims, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnRollGetWorkspaceSize(acl_src, acl_shifts, acl_dims, acl_dst,
                                        &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnRoll(workspaceAddr, workspaceSize, executor, ctx.stream());

    aclDestroyIntArray(acl_shifts);
    aclDestroyIntArray(acl_dims);
    aclrtFree(workspaceAddr);
}

static void aclnn_index_fill_tensor(ggml_backend_ascend_context& ctx,
                                    aclTensor* acl_src, int64_t dim,
                                    int64_t* index, int64_t index_num,
                                    float value) {
    aclIntArray* acl_index = aclCreateIntArray(index, index_num);
    aclScalar* acl_value = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnInplaceIndexFillTensorGetWorkspaceSize(
        acl_src, dim, acl_index, acl_value, &workspaceSize, &executor);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnnInplaceIndexFillTensor(workspaceAddr, workspaceSize,
                                          executor, ctx.stream());

    aclDestroyIntArray(acl_index);
    aclDestroyScalar(acl_value);
    aclrtFree(workspaceAddr);
}

void ggml_ascend_rope_new(ggml_backend_ascend_context& ctx, ggml_tensor* dst) {
    // TODO: use ascendc
    // Only test with LLAMA model.
    ggml_tensor* src0 = dst->src[0];  // input
    ggml_tensor* src2 = dst->src[2];  // freq_factors

    // TODO: with freq_factors
    GGML_ASSERT(src2 == NULL);

    // param
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims = ((int32_t*)dst->op_params)[1];
    const int mode = ((int32_t*)dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t*)dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    memcpy(&freq_base, (int32_t*)dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t*)dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t*)dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t*)dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t*)dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t*)dst->op_params + 10, sizeof(float));

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // TODO: ext_factor != 0
    GGML_ASSERT(ext_factor == 0);
    // TODO: freq_scale != 1
    GGML_ASSERT(freq_scale == 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims);

    const bool is_neox = mode & 2;

    // init cos/sin cache
    void* sin_buffer(nullptr);
    void* cos_buffer(nullptr);
    aclrtMalloc(&sin_buffer, src0->ne[0] * src0->ne[2] * sizeof(float_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&cos_buffer, src0->ne[0] * src0->ne[2] * sizeof(float_t), ACL_MEM_MALLOC_NORMAL_ONLY);

    int64_t sin_reshape_ne[4] = {src0->ne[0], 1, src0->ne[2], 1};
    size_t sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = sizeof(float_t);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    aclTensor* acl_sin_reshape_tensor =
        ggml_ascend_create_tensor(sin_buffer, ACL_FLOAT, sizeof(float_t),
                                sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    aclTensor* acl_cos_reshape_tensor =
        ggml_ascend_create_tensor(cos_buffer, ACL_FLOAT, sizeof(float_t),
                                sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    // aclnn_cache_init(ctx, dst, acl_cos_reshape_tensor, acl_sin_reshape_tensor,
    //                  theta_scale, is_neox);

    // roll input
    void* input_roll_buffer;
    aclTensor* acl_minus_one_tensor;
    void* minus_one_scale_buffer = nullptr;

    void *roll_ptr;
    void *scale_ptr;
    aclrtMalloc(&roll_ptr, ggml_nbytes(src0), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&scale_ptr, sizeof(float_t) * src0->ne[0], ACL_MEM_MALLOC_NORMAL_ONLY);

    if (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        input_roll_buffer = roll_ptr;
        int64_t input_roll_ne[4] = {2, src0->ne[1] * (src0->ne[0] / 2),
                                    src0->ne[2], src0->ne[3]};
        size_t input_roll_nb[GGML_MAX_DIMS];
        input_roll_nb[0] = ggml_type_size(src0->type);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_roll_nb[i] = input_roll_nb[i - 1] * input_roll_ne[i - 1];
        }
        aclTensor* acl_input_roll_tensor = ggml_ascend_create_tensor(
            input_roll_buffer, ggml_to_acl_map[src0->type],
            ggml_type_size(src0->type), input_roll_ne, input_roll_nb,
            GGML_MAX_DIMS);
        aclTensor* acl_input_tensor = ggml_ascend_create_tensor(
            src0->data, ggml_to_acl_map[src0->type],
            ggml_type_size(src0->type), input_roll_ne, input_roll_nb,
            GGML_MAX_DIMS);

        int64_t shifts[] = {1};
        int64_t dims[] = {3};
        aclnn_roll(ctx, acl_input_tensor, acl_input_roll_tensor, shifts, dims);
        aclDestroyTensor(acl_input_roll_tensor);
        aclDestroyTensor(acl_input_tensor);

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_buffer = scale_ptr;

        int64_t minus_one_ne[4] = {src0->ne[0], 1, 1, 1};
        size_t minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_ones(
            ctx, minus_one_scale_buffer, sizeof(float_t) * src0->ne[0],
            minus_one_ne, GGML_MAX_DIMS, ACL_FLOAT, sizeof(float_t), 1);
        int64_t dim = 3;
        int64_t* index = new int64_t[src0->ne[0]];
        for (int i = 0; i < src0->ne[0]; i++) {
            index[i] = i / 2 * 2;
        }
        int64_t index_num = src0->ne[0];
        float value = -1;
        aclnn_index_fill_tensor(ctx, acl_minus_one_tensor, dim, index,
                                index_num, value);
    } else {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        input_roll_buffer = roll_ptr;
        aclTensor* acl_input_roll_tensor = ggml_ascend_create_tensor(
            input_roll_buffer, ggml_to_acl_map[src0->type],
            ggml_type_size(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);
        aclTensor* acl_input_tensor = ggml_ascend_create_tensor(src0);

        int64_t shifts[] = {src0->ne[0] / 2};
        int64_t dims[] = {3};
        aclnn_roll(ctx, acl_input_tensor, acl_input_roll_tensor, shifts, dims);

        aclDestroyTensor(acl_input_roll_tensor);
        aclDestroyTensor(acl_input_tensor);

        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_buffer = scale_ptr;

        int64_t minus_one_ne[4] = {src0->ne[0], 1, 1, 1};
        size_t minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_ones(
            ctx, minus_one_scale_buffer, sizeof(float_t) * src0->ne[0],
            minus_one_ne, GGML_MAX_DIMS, ACL_FLOAT, sizeof(float_t), 1);
        // -1 * first half
        int64_t first_half_ne[4] = {src0->ne[0] / 2, 1, 1, 1};
        size_t first_half_nb[GGML_MAX_DIMS];
        first_half_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            first_half_nb[i] = first_half_nb[i - 1] * first_half_ne[i - 1];
        }
        aclTensor* acl_first_half_tensor = ggml_ascend_create_tensor(
            minus_one_scale_buffer, ACL_FLOAT, sizeof(float_t), first_half_ne,
            first_half_nb, GGML_MAX_DIMS);
        bool inplace = true;
        float scale = -1;
        aclnn_muls(ctx, acl_first_half_tensor, scale, nullptr, inplace);
        aclDestroyTensor(acl_first_half_tensor);
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    void* input_roll_mul_scale_buffer(nullptr);
    aclrtMalloc(&input_roll_mul_scale_buffer, ggml_nbytes(src0), ACL_MEM_MALLOC_NORMAL_ONLY);
    size_t input_nb[GGML_MAX_DIMS];
    input_nb[0] = ggml_type_size(src0->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        input_nb[i] = input_nb[i - 1] * src0->ne[i - 1];
    }
    aclTensor* acl_input_roll_mul_scale_tensor = ggml_ascend_create_tensor(
        input_roll_mul_scale_buffer, ggml_to_acl_map[src0->type],
        ggml_type_size(src0->type), src0->ne, input_nb, GGML_MAX_DIMS);
    aclTensor* acl_input_roll_reshape_tensor = ggml_ascend_create_tensor(
        input_roll_buffer, ggml_to_acl_map[src0->type],
        ggml_type_size(src0->type), src0->ne, input_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_input_roll_reshape_tensor, acl_minus_one_tensor,
              acl_input_roll_mul_scale_tensor);

    // output
    aclTensor* acl_src0 = ggml_ascend_create_tensor(src0);
    aclTensor* acl_dst = ggml_ascend_create_tensor(dst);
    void* output_fp32_buffer;
    if (src0->type == GGML_TYPE_F32) {
        aclnn_inplace_mul(ctx, acl_src0, acl_cos_reshape_tensor);
        aclnn_inplace_mul(ctx, acl_input_roll_mul_scale_tensor,
                          acl_sin_reshape_tensor);
        ggml_ascend_add_fn(ctx, acl_src0, acl_input_roll_mul_scale_tensor, acl_dst);
        // TODO: ne0 != n_dims in mode2
    } else if (src0->type == GGML_TYPE_F16) {
        size_t input_fp32_nb[GGML_MAX_DIMS];
        input_fp32_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_fp32_nb[i] = input_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        
        void* input_fp32_buffer1(nullptr);
        aclrtMalloc(&input_fp32_buffer1, ggml_nelements(dst) * sizeof(float_t), ACL_MEM_MALLOC_NORMAL_ONLY);
        aclTensor* input_fp32_tensor1 = ggml_ascend_create_tensor(
            input_fp32_buffer1, ACL_FLOAT, sizeof(float_t), dst->ne,
            input_fp32_nb, GGML_MAX_DIMS);

        void* input_fp32_buffer2(nullptr);
        aclrtMalloc(&input_fp32_buffer2, ggml_nelements(dst) * sizeof(float_t), ACL_MEM_MALLOC_NORMAL_ONLY);
        aclTensor* input_fp32_tensor2 = ggml_ascend_create_tensor(
            input_fp32_buffer2, ACL_FLOAT, sizeof(float_t), dst->ne,
            input_fp32_nb, GGML_MAX_DIMS);

        // output_fp32_buffer = fp32_allocator.get();
        aclrtMalloc(&output_fp32_buffer, ggml_nelements(dst) * sizeof(float_t), ACL_MEM_MALLOC_NORMAL_ONLY);
        aclTensor* output_fp32_tensor = ggml_ascend_create_tensor(
            output_fp32_buffer, ACL_FLOAT, sizeof(float_t), dst->ne,
            input_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src0, acl_cos_reshape_tensor, input_fp32_tensor1);
        aclnn_mul(ctx, acl_input_roll_mul_scale_tensor, acl_sin_reshape_tensor,
                  input_fp32_tensor2);
        ggml_ascend_add_fn(ctx, input_fp32_tensor1, input_fp32_tensor2,
                  output_fp32_tensor);
        aclnn_cast(ctx, output_fp32_tensor, acl_dst, ACL_FLOAT16);

        aclDestroyTensor(input_fp32_tensor1);
        aclDestroyTensor(input_fp32_tensor2);
        aclDestroyTensor(output_fp32_tensor);

        aclrtFree(input_fp32_buffer1);
        aclrtFree(input_fp32_buffer2);
        aclrtFree(output_fp32_buffer);

    }

    aclDestroyTensor(acl_sin_reshape_tensor);
    aclDestroyTensor(acl_cos_reshape_tensor);
    aclDestroyTensor(acl_minus_one_tensor);
    aclDestroyTensor(acl_input_roll_mul_scale_tensor);
    aclDestroyTensor(acl_input_roll_reshape_tensor);
    aclDestroyTensor(acl_src0);
    aclDestroyTensor(acl_dst);

    aclrtFree(sin_buffer);
    aclrtFree(cos_buffer);
    aclrtFree(roll_ptr);
    aclrtFree(scale_ptr);
    aclrtFree(input_roll_mul_scale_buffer);
}