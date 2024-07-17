#include <iostream>
#include <vector>
#include "common.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_hardsigmoid.h"
#include "aclnn-unary.h"
#include "aclnn-cpy.h"
#include "aclnn-add.h"
#include "aclnn-mul.h"
#include "aclnn-compute.h"
#include "aclnn-test.h"

void aclnn_silu_test_all(){
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtContext context = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));

    int64_t lens[3]{2, 3, 4};
    int64_t width[3]{1, 2, 3};
    float data1[2]{-5.1, -3.7};
    float data2[6]{0.1, 0.6, 1.1, 1.6, 2.1, 2.6};
    float data3[12]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};
    ggml_ascend_silu_test(lens[0], width[0], data1, deviceId, stream);
    ggml_ascend_silu_test(lens[1], width[1], data2, deviceId, stream);
    ggml_ascend_silu_test(lens[2], width[2], data3, deviceId, stream);
    aclnn_silu_func_test(lens[0], width[0], data1, deviceId, stream);
    aclnn_silu_func_test(lens[1], width[1], data2, deviceId, stream);
    aclnn_silu_func_test(lens[2], width[2], data3, deviceId, stream);
}


void aclnn_dup_test_all(){
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtContext context = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));

    int64_t ne1[4] = {1, 2, 3, 4};
    int64_t ne2[4] = {3, 3, 3, 2};
    float data1[24] = {1.23, 4.56, 7.89, 0.12, 3.45, 6.78, 9.01, 2.34, 5.67, 8.90, 1.11, 4.22, 7.33, 0.44, 3.55, 6.66, 9.77, 2.88, 5.99, 1.01, 4.12, 7.23, 0.34, 3.45};
    float data2[54] = {0.5, 1.2, 3.7, 2.8, 4.1, 5.6, 6.9, 7.2, 8.5, 9.3, 10.7, 11.4, 12.8, 13.5, 14.9, 15.6, 16.3, 17.8, 18.5, 19.2, 20.7, 21.4, 22.9, 23.6, 24.3, 25.8, 26.5, 27.2, 28.7, 29.4, 30.9, 31.6, 32.3, 33.8, 34.5, 35.2, 36.7, 37.4, 38.9, 39.6, 40.3, 41.8, 42.5, 43.2, 44.7, 45.4, 46.9, 47.6, 48.3, 49.8, 50.5, 51.2, 52.7, 53.4};
    ggml_ascend_dup_test(ne1, data1, deviceId, stream);
    ggml_ascend_dup_test(ne2, data2, deviceId, stream);
}

void aclnn_cpy_test_all(){
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtContext context = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));

    int64_t ne1[4] = {1, 2, 3, 4};
    int64_t ne2[4] = {3, 3, 3, 2};
    float data1[24] = {1.23, 4.56, 7.89, 0.12, 3.45, 6.78, 9.01, 2.34, 5.67, 8.90, 1.11, 4.22, 7.33, 0.44, 3.55, 6.66, 9.77, 2.88, 5.99, 1.01, 4.12, 7.23, 0.34, 3.45};
    float data2[54] = {0.5, 1.2, 3.7, 2.8, 4.1, 5.6, 6.9, 7.2, 8.5, 9.3, 10.7, 11.4, 12.8, 13.5, 14.9, 15.6, 16.3, 17.8, 18.5, 19.2, 20.7, 21.4, 22.9, 23.6, 24.3, 25.8, 26.5, 27.2, 28.7, 29.4, 30.9, 31.6, 32.3, 33.8, 34.5, 35.2, 36.7, 37.4, 38.9, 39.6, 40.3, 41.8, 42.5, 43.2, 44.7, 45.4, 46.9, 47.6, 48.3, 49.8, 50.5, 51.2, 52.7, 53.4};
    aclnn_cpy_func_test(ne1, data1, deviceId, stream);
    aclnn_cpy_func_test(ne2, data2, deviceId, stream);
    ggml_ascend_cpy_test(ne1, data1, deviceId, stream);
    ggml_ascend_cpy_test(ne2, data2, deviceId, stream);
}


void aclnn_mul_test_all(){
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtContext context = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));

    int64_t ne1[4] = {1, 2, 2, 2};
    int64_t ne2[4] = {2, 1, 2, 2};
    float data1[8] = {0.5, 1.2, 2.7, 3.1, 4.6, 5.0, 6.3, 7.9};
    float data2[8] = {3.14, 2.71, 1.41, 0.57, 4.67, 8.99, 5.23, 6.28};
    aclnn_mul_func_test(ne1, ne2, data1, data2, deviceId, stream);
    ggml_ascend_mul_test(ne1, ne2, data1, data2, deviceId, stream); 
}

void aclnn_add_test_all(){
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtContext context = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));

    int64_t ne1[4] = {1, 2, 2, 2};
    int64_t ne2[4] = {2, 1, 2, 2};
    float data1[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    float data2[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    aclnn_add_func_test(ne1, ne2, data1, data2, deviceId, stream);
    ggml_ascend_add_test(ne1, ne2, data1, data2, deviceId, stream);
}

void aclnn_get_rows_test_all(){
    int32_t deviceId = 0;
    aclrtStream stream;
    aclrtContext context = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret));

    int64_t ne1[4] = {3, 2, 2, 2};
    int64_t ne2[4] = {3, 2, 2, 1};
    float data1[24] = {
        0.0, 1.0, 2.0,      3.0, 4.0, 5.0,
        6.0, 7.0, 8.0,      9.0, 10.0, 11.0,

        12.0, 13.0, 14.0,   15.0, 16.0, 17.0,
        18.0, 19.0, 20.0,   21.0, 22.0, 23.0
    };
    int64_t data2[12] = {
        0, 1, 1,  1, 1, 0,
        0, 1, 0,  1, 1, 1,
    };
    ggml_ascend_get_rows_test(ne1, ne2, data1, data2, deviceId, stream);
}