//
// Created by 35763 on 2024/6/26.
//
#pragma once

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <memory>

#include "ggml.h"
#include "ggml-ascend.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

#define CHECK_RET(cond, return_expr) \
do {                               \
if (!(cond)) {                   \
return_expr;                   \
}                                \
} while (0)

#define LOG_PRINT(message, ...)     \
do {                              \
printf(message, ##__VA_ARGS__); \
} while (0)

#define GGML_ASCEND_MAX_STREAMS 8

#define aclnn_shape_t std::vector<int64_t>

extern aclDataType ggml_to_acl_map[GGML_TYPE_COUNT];

int64_t GetShapeSize(const std::vector<int64_t>& shape);
int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream);

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int create_acl_tensor(const aclnn_shape_t& shape, aclDataType dataType, void** deviceAddr, aclTensor** tensor);

void ggml_ascend_set_device(int device);
int ggml_ascend_get_device();

struct ggml_ascend_pool {
    virtual ~ggml_ascend_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
};

template <typename T>
int data_addr_malloc(const aclnn_shape_t& shape, const std::vector<T>& hostData, void** deviceAddr) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template<typename T>
struct ggml_ascend_pool_alloc {
    ggml_ascend_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_ascend_pool_alloc() = default;

    explicit ggml_ascend_pool_alloc(ggml_ascend_pool & pool) : pool(&pool) {
    }

    ggml_ascend_pool_alloc(ggml_ascend_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_ascend_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_ascend_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_ascend_pool_alloc(const ggml_ascend_pool_alloc &) = delete;
    ggml_ascend_pool_alloc(ggml_ascend_pool_alloc &&) = delete;
    ggml_ascend_pool_alloc& operator=(const ggml_ascend_pool_alloc &) = delete;
    ggml_ascend_pool_alloc& operator=(ggml_ascend_pool_alloc &&) = delete;
};

struct ggml_ascend_graph {

};

struct ggml_backend_ascend_context {
    int device;
    std::string name;
    aclrtEvent event = nullptr;

    aclrtStream streams[GGML_ASCEND_MAX_DEVICES][GGML_ASCEND_MAX_STREAMS] = { { nullptr } };
    void * opHandles[GGML_ASCEND_MAX_DEVICES] = { nullptr }; //??

    std::unique_ptr<ggml_ascend_graph> ascend_graph;

    explicit ggml_backend_ascend_context(int device) :
        device(device),
        name(GGML_ASCEND_NAME + std::to_string(device)) {}

    ~ggml_backend_ascend_context() {
        if (event != nullptr) {
            // todo Check: fixed
            auto ret = aclrtDestroyEvent(event);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl destroy event failed at [~ggml_backend_ascend_context]: %d\n", ret));
        }

        for (auto i = 0; i < GGML_ASCEND_MAX_DEVICES; ++i) {
            for (auto j = 0; j < GGML_ASCEND_MAX_STREAMS; ++j) {
                if (streams[i][j] != nullptr) {
                    // todo Check: fixed
                    auto ret = aclrtDestroyStream(streams[i][j]);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl destroy stream failed at [~ggml_backend_ascend_context]: %d\n", ret));
                }
            }

            if (opHandles[i] != nullptr) {
                // todo Check: fixed
            }
        }
    }

    void ggml_ascend_set_device(int device) {
        int current_device;
        // todo Check: fixed
        auto ret = aclrtGetDevice(&current_device);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl get device failed at [ggml_ascend_set_device]: %d\n", ret); return);

        if (device == current_device) {
            return;
        }

        // todo Check: fixed
        ret = aclrtSetDevice(device);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl set device failed at [ggml_ascend_set_device]: %d\n", ret); return);
    }

    aclrtStream stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_ascend_set_device(device);
            // GGML_UNUSED(); todo check
            aclrtCreateStreamWithConfig(&streams[device][stream], 0, ACL_STREAM_FAST_SYNC);
        }
        return streams[device][stream];
    }

    aclrtStream stream() {
        return stream(device, 0);
    }

    std::unique_ptr<ggml_ascend_pool> pools[GGML_ASCEND_MAX_DEVICES];

    static std::unique_ptr<ggml_ascend_pool> new_pool_for_device(int device);

    ggml_ascend_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }

    ggml_ascend_pool & pool() {
        return pool(device);
    }
};

#endif //COMMON_H
