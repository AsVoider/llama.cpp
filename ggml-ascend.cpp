#include "ggml.h"
#include "ggml-ascend.h"
#include "ggml-backend-impl.h"

// Fixme: This file including is not suitable
#include "acl/acl.h"

#include "ggml-Ascend/common.h"
#include "ggml-Ascend/aclnn-add.h"
#include "ggml-Ascend/aclnn-arange.h"
#include "ggml-Ascend/aclnn-clamp.h"
#include "ggml-Ascend/aclnn-comp.h"
#include "ggml-Ascend/aclnn-div.h"
#include "ggml-Ascend/aclnn-im2col.h"
#include "ggml-Ascend/aclnn-leaky.h"
#include "ggml-Ascend/aclnn-mul.h"
#include "ggml-Ascend/aclnn-norm.h"
#include "ggml-Ascend/aclnn-operator.h"
#include "ggml-Ascend/aclnn-rope.h"
#include "ggml-Ascend/aclnn-sort.h"
#include "ggml-Ascend/aclnn-unary.h"
#include "ggml-Ascend/aclnn-compute.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define MATRIX_ROW_PADDING 512

static void ggml_ascend_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fprintf(stderr, "%s", msg);
}

ggml_log_callback ggml_ascend_log_callback = ggml_ascend_default_log_callback;
void * ggml_ascend_log_user_data = NULL;

GGML_API void ggml_backend_ascend_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_ascend_log_callback = log_callback;
    ggml_ascend_log_user_data = user_data;
}

#define GGML_ASCEND_LOG_INFO(...) ggml_ascend_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_ASCEND_LOG_WARN(...) ggml_ascend_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_ASCEND_LOG_ERROR(...) ggml_ascend_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_ascend_log(enum ggml_log_level level, const char * format, ...) {
    if (ggml_ascend_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_ascend_log_callback(level, buffer, ggml_ascend_log_user_data);
        } else {
            std::vector<char> buffer2(len + 1);
            va_end(args);
            va_start(args, format);
            vsnprintf(&buffer2[0], buffer2.size(), format, args);
            ggml_ascend_log_callback(level, buffer2.data(), ggml_ascend_log_user_data);
        }
        va_end(args);
    }
}

[[noreturn]]
void ggml_ascend_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1;
    aclrtGetDevice(&id);

    GGML_ASCEND_LOG_ERROR("ASCEND error: %s\n", msg);
    GGML_ASCEND_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_ASCEND_LOG_ERROR("  %s\n", stmt);

    GGML_ASSERT(!"ASCEND error");
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

int ggml_ascend_get_device() {
    int id;
    // todo check: fixed
    auto ret = aclrtGetDevice(&id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl get device failed at [ggml_ascend_get_device]: %d\n", ret); return -1);
    return id;
}

static aclError ggml_ascend_device_malloc(void ** ptr, size_t size, int device) {
    ggml_ascend_set_device(device);

    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

//buffer pool
struct ggml_ascend_pool_leg : public ggml_ascend_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_ascend_buffer {
        void *ptr;
        size_t size = 0;
    };

    ggml_ascend_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_ascend_pool_leg(int device) : device(device) {

    }

    ~ggml_ascend_pool_leg() {
        ggml_ascend_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            if (auto &b = buffer_pool[i]; b.ptr != nullptr) {
                // todo Check: fixed
                auto ret = aclrtFree(b.ptr);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl free failed at [~ggml_ascend_pool_leg]: %d\n", ret); return);
                pool_size -= b.size;
            }
            GGML_ASSERT(pool_size == 0);
            // todo assert: fixed
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        size_t best_diff = 1ull << 36;  // 2 * 36
        int ibest = 1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            if (auto &b = buffer_pool[i]; b.ptr != nullptr) {
                if (b.size >= size) {
                    auto diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            auto &b = buffer_pool[ibest];
            auto ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        auto look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
        ggml_ascend_set_device(device);
        // todo Check: fixed
        auto ret = ggml_ascend_device_malloc(&ptr, look_ahead_size, device);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl device malloc failed at [pool_leg malloc]: %d\n", ret); return nullptr);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; i++) {
            auto &b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_ASCEND_LOG_WARN("ASCEND buffer pool full, increase MAX_BUFFERS\n");
        ggml_ascend_set_device(device);
        // todo Check: fixed
        auto ret = aclrtFree(ptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl free failed at [ggml_ascend_pool_leg free]: %d\n", ret); return);
        pool_size -= size;
    }
}

std::unique_ptr<ggml_ascend_pool> ggml_backend_ascend_context::new_pool_for_device(int device) {
    // todo if vmm?: fixed
    return std::unique_ptr<ggml_ascend_pool>(new ggml_ascend_pool_leg(device));
}

// ascend buffer copy

struct ggml_backend_ascend_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_ascend_buffer_context() :
        device(device), dev_ptr(dev_ptr), 
        name(GGML_ASCEND_NAME + std::to_string(device)) {
    } 

    ~ggml_backend_ascend_buffer_context() {
        // todo Check: fixed
        auto ret = aclrtFree(dev_ptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl free failed at [~ggml_backend_ascend_buffer_context]: %d\n", ret); return);
    }
};

GGML_CALL static const char * ggml_backend_ascend_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_ascend_buffer_context * ctx = (ggml_backend_ascend_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_ascend(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_ascend_buffer_get_name;
}

GGML_CALL static void ggml_backend_ascend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_ascend_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_ascend_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto ctx = (ggml_backend_ascend_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_ascend_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_ascend_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type)) {
        auto original_size = ggml_nbytes(tensor);
        auto padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            ggml_ascend_set_device(ctx->device);
            // todo Check: fixed
            auto ret = aclrtMemset((char *)tensor->data + original_size, padded_size - original_size, 0, padded_size - original_size);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memset failed at [ggml_backend_ascend_buffer_init_tensor]: %d\n", ret); return);
        }
    }
}

GGML_CALL static void ggml_backend_ascend_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto ctx = (ggml_backend_ascend_buffer_context *)buffer->context;

    ggml_ascend_set_device(ctx->device);

    // todo Check: fixed
    auto ret = aclrtMemcpy((char *)tensor->data, offset, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpy failed at [ggml_backend_ascend_buffer_set_tensor]: %d\n", ret); return);
}

GGML_CALL static void ggml_backend_ascend_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto ctx = (ggml_backend_ascend_buffer_context *)buffer->context;

    ggml_ascend_set_device(ctx->device);
    //todo Check: fixed
    auto ret = aclrtMemcpy(data, size, (const char *)tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpy failed at [ggml_backend_ascend_buffer_get_tensor]: %d\n", ret); return);
}

GGML_CALL static bool ggml_backend_ascend_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_ascend(src->buffer)) {
        auto src_ctx = (ggml_backend_ascend_buffer_context *)src->buffer->context;
        auto dst_ctx = (ggml_backend_ascend_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            // todo Check: fixed
            // Now we don't support peer copy
            auto ret = aclrtMemcpy(dst->data, ggml_nbytes(src), src->data, ggml_nbytes(src), ACL_MEMCPY_DEVICE_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpy failed at [ggml_backend_ascend_buffer_cpy_tensor]: %d\n", ret); return false);
        } else {
            return false;
        }
        return true;
    }

    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_ascend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto ctx = (ggml_backend_ascend_buffer_context *)buffer->context;

    ggml_ascend_set_device(ctx->device);
    // todo Check: fixed
    auto ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl sync1 failed at [ggml_backend_ascend_buffer_clear]: %d\n", ret); return);
    ret = aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memset failed at [ggml_backend_ascend_buffer_clear]: %d\n", ret); return);
    ret = aclrtSynchronizeDevice();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl sync2 failed at [ggml_backend_ascend_buffer_clear]: %d\n", ret); return);
}

static ggml_backend_buffer_i ggml_backend_ascend_buffer_interface = {
    /* .get_name    = */ ggml_backend_ascend_buffer_get_name,
    /* .free_buffer = */ ggml_backend_ascend_buffer_free_buffer,
    /* .get_base    = */ ggml_backend_ascend_buffer_get_base,
    /* .init_tensor = */ ggml_backend_ascend_buffer_init_tensor,
    /* .set_tensor  = */ ggml_backend_ascend_buffer_set_tensor,
    /* .get_tensor  = */ ggml_backend_ascend_buffer_get_tensor,
    /* .cpy_tensor  = */ ggml_backend_ascend_buffer_cpy_tensor,
    /* .clear       = */ ggml_backend_ascend_buffer_clear,
    /* .reset       = */ NULL,
};

struct ggml_backend_ascend_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_ascend_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto ctx = (ggml_backend_ascend_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_ascend(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_ascend_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_ascend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto buft_ctx = (ggml_backend_ascend_buffer_type_context *)buft->context;

    ggml_ascend_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1);

    void * dev_ptr;
    auto err = ggml_ascend_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != ACL_SUCCESS) {
        // todo log: fixed
        GGML_ASCEND_LOG_ERROR("%s, alloc %.2f MiB on device %d: aclrtMalloc failed: %d\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, err);
        return nullptr;
    }

    auto ctx = new ggml_backend_ascend_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_ascend_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_ascend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128; // todo REALLY?: fixed just now

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_ascend_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    auto size = ggml_nbytes(tensor);
    auto ne0 = tensor->ne[0];

    // not support quantize ?
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_ascend_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_ascend_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_ascend_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_ascend_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_ascend_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_ascend_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_ascend_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_ascend_buffer_types[GGML_ASCEND_MAX_DEVICES];

    static bool ggml_backend_ascend_buffer_type_initialized = false;

    if (!ggml_backend_ascend_buffer_type_initialized) {
        for (auto i = 0; i < GGML_ASCEND_MAX_DEVICES; i++) {
            ggml_backend_ascend_buffer_types[i] = {
                /* .iface   */ ggml_backend_ascend_buffer_type_interface,
                /* .context */ new ggml_backend_ascend_buffer_type_context{i, GGML_ASCEND_NAME + std::to_string(i)},

                // todo implement！！！！！！！！！ fixed
            };
        }
        ggml_backend_ascend_buffer_type_initialized = true;
    }

    return &ggml_backend_ascend_buffer_types[device];
}

GGML_CALL int ggml_backend_ascend_get_device_count() {
    uint32_t ret = 0;
    auto err = aclrtGetDeviceCount(&ret);
    return err == ACL_SUCCESS ? ret : 0;
}

//---------------------- todo split --------------------- //

GGML_CALL static const char * ggml_backend_ascend_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_ASCEND_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_ascend_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_ASCEND_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_ascend_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    //todo CHECK: fixed
    auto ret = aclrtFreeHost(buffer->context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl free host failed at [ggml_backend_ascend_host_buffer_free_buffer]: %d\n", ret); return);
}

static void *  ggml_ascend_host_malloc(size_t size) {
    if (getenv("GGML_ASCEND_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    aclError err = aclrtMallocHost((void **) &ptr, size);
    if (err != ACL_SUCCESS) {
        //todo: ERROR LOG: fixed
        GGML_ASCEND_LOG_ERROR("%s, failed to allocate %.2f MiB of pinned memory: %d\n", __func__,
                              size / 1024.0 / 1024.0, err);
        return nullptr;
    }

    return ptr;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_ascend_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_ascend_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_ascend_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_ascend_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_ascend_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_ascend_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_ascend_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_ascend_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_ascend_buffer_type_host;
}

// ------------------------ todo kernels ----------------------- //


// ------------------------ todo peers ------------------------- //


GGML_CALL void ggml_backend_ascend_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_ascend_set_device(device);

    // todo Check: fixed
    auto ret = aclrtGetMemInfo(ACL_DDR_MEM, free, total);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl GetMemInfo failed at [ggml_backend_ascend_get_device_memory]: %d\n", ret); return);
}

//////////////////////////////////////////////////////////////////////////////////
// todo compute!!!
static bool ggml_ascend_compute_forward(ggml_backend_ascend_context & ctx, struct ggml_tensor * dst) {
    // mark: no peer access
    switch (dst->op) {
        case GGML_OP_GET_ROWS:
            ggml_ascend_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_ascend_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_ascend_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_ascend_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_ascend_add(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_ascend_mul(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_SILU:
                    ggml_ascend_silu(ctx, dst);
                    break;
                default:
                    return false;
            }   
            break;
        case GGML_OP_RMS_NORM:
            ggml_ascend_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                GGML_ASCEND_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return false;
            } else {
                ggml_ascend_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            }
            break;
        case GGML_OP_SOFT_MAX:
            ggml_ascend_soft_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_ascend_rope(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

//////////////////////////////////////////////////////////////////////////////////
// TODO BACKEND!!!!

GGML_CALL static const char * ggml_backend_ascend_name(ggml_backend_t backend) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;

    return ctx->name.c_str();
}

GGML_CALL static void ggml_backend_ascend_free(ggml_backend_t backend) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;

    delete ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_ascend_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_ascend_context *)backend->context;

    return ggml_backend_ascend_buffer_type(ctx->device);
}

GGML_CALL static void ggml_backend_ascend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;
    auto buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_ascend_buffer_type(ctx->device) && "unsupport buffer type");
    // todo Check
    auto ret = aclrtMemcpyAsync((char *)tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE, ctx->stream());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpyAsync failed at [ggml_backend_ascend_set_tensor_async]: %d\n", ret); return);
}

GGML_CALL static void ggml_backend_ascend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;
    auto buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_ascend_buffer_type(ctx->device) && "unsupported buffer type");
    // todo Check
    auto ret = aclrtMemcpyAsync(data, size, (const char *)tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST, ctx->stream());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpyAsync failed at [ggml_backend_ascend_get_tensor_async]: %d\n", ret); return);
}

// dangerous
GGML_CALL static bool ggml_backend_ascend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_backend_is_ascend(backend_src) || ggml_backend_is_ascend(backend_dst));

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_buffer_is_ascend(src->buffer)) {
        return false;
    }

    if (!ggml_backend_buffer_is_ascend(dst->buffer)) {
        return false;
    }

    // device -> device
    ggml_backend_ascend_context * ctx_src = (ggml_backend_ascend_context *)backend_src->context;
    ggml_backend_ascend_context * ctx_dst = (ggml_backend_ascend_context *)backend_dst->context;

    if (backend_src != backend_dst) {
        ggml_backend_ascend_buffer_context * buf_ctx_src = (ggml_backend_ascend_buffer_context *)buf_src->context;
        ggml_backend_ascend_buffer_context * buf_ctx_dst = (ggml_backend_ascend_buffer_context *)buf_dst->context;

        GGML_ASSERT(ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(ctx_dst->device == buf_ctx_dst->device);

        // copy on src stream
        if (ctx_src->device == ctx_dst->device) {
            auto ret = aclrtMemcpyAsync(dst->data, ggml_nbytes(dst), src->data, ggml_nbytes(dst), ACL_MEMCPY_DEVICE_TO_DEVICE, ctx_dst->stream());
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpyAsync1 failed at [ggml_backend_ascend_cpy_tensor_async]: %d\n", ret); return false);
        } else {
            // todo Not Support Peer Copy: marked
            return false;
        }

        // record event on src stream
        if (!ctx_src->event) {
            ggml_ascend_set_device(ctx_src->device);
            // todo FLAG CHECK: default
            auto ret = aclrtCreateEvent(&ctx_src->event);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl create event failed at [ggml_backend_ascend_cpy_tensor_async]: %d\n", ret); return false);
        }

        // todo check: fixed
        auto ret = aclrtRecordEvent(ctx_src->event, ctx_src->stream());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl record event failed at [ggml_backend_ascend_cpy_tensor_async]: %d\n", ret); return false);

        // wait on dst stream for the copy to complete
        // todo check: fixed
        ret = aclrtStreamWaitEvent(ctx_dst->stream(), ctx_src->event);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl stream wait event failed at [ggml_backend_ascend_cpy_tensor_async]: %d\n", ret); return false);
    } else {
        // src and dst are on the same backend
        // todo check: fixed
        auto ret = aclrtMemcpyAsync(dst->data, ggml_nbytes(dst), src->data, ggml_nbytes(dst), ACL_MEMCPY_DEVICE_TO_DEVICE, ctx_dst->stream());
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl memcpyAsync2 failed at [ggml_backend_ascend_cpy_tensor_async]: %d\n", ret); return false);
    }
    return true;
}

GGML_CALL static void ggml_backend_ascend_synchronize(ggml_backend_t backend) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;

    //todo Check: fixed
    auto ret = aclrtSynchronizeStream(ctx->stream());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl SynchronizeStream failed at [ggml_backend_ascend_synchronize]: %d\n", ret));

    GGML_UNUSED(backend);
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address &&
          node->op != GGML_OP_CPY &&
          node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != graph_node_properties->src_address[i] &&
            node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW
        ) {
            return false;
        }
    }
    return true;
}

GGML_CALL static enum ggml_status ggml_backend_ascend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;
    ggml_ascend_set_device(ctx->device);

    bool use_graph = false;
    bool graph_update_required = false;

    bool graph_evaluated_or_captured = false;

    while (!graph_evaluated_or_captured) {
        if (!use_graph || graph_evaluated_or_captured) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }
#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_ascend_buffer_type(ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer->buft == ggml_backend_ascend_buffer_type(ctx->device));
                    }
                }
#endif
                // TODO implement: marked
                bool ok = ggml_ascend_compute_forward(*ctx, node);
                if (!ok) {
                    // todo LOG: fixed
                    GGML_ASCEND_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }
        }
        graph_evaluated_or_captured = true;
    }

    return GGML_STATUS_SUCCESS;
}

// todo !!!!! marked: fixed
GGML_CALL static bool ggml_backend_ascend_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    auto ascend_ctx = (ggml_backend_ascend_context *)backend->context;
    switch (op->op) {
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(op)) {
            case GGML_UNARY_OP_SILU:
                return ggml_is_contiguous(op->src[0]);
            default:
                return false;
        }
        break;
    case GGML_OP_MUL_MAT:
        {
            struct ggml_tensor * a;
            struct ggml_tensor * b;
            if (op->op == GGML_OP_MUL_MAT) {
                a = op->src[0];
                b = op->src[1];
            }
            if (a->ne[3] != b->ne[3]) {
                return false;
            }
            ggml_type a_type = a->type;
            if (a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ3_XXS ||
                a_type == GGML_TYPE_IQ1_S   || a_type == GGML_TYPE_IQ4_NL || a_type == GGML_TYPE_IQ3_S   ||
                a_type == GGML_TYPE_IQ1_M   || a_type == GGML_TYPE_IQ2_S  || a_type == GGML_TYPE_IQ4_XS) {
                if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
                    return false;
                }
            }
            return true;
        } break;
    case GGML_OP_GET_ROWS:
        {
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        } break;
    case GGML_OP_CPY:
        {
            ggml_type src0_type = op->src[0]->type;
            ggml_type src1_type = op->src[1]->type;
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                return true;
            }
            if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                return true;
            }
            if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                 return true;
            }
            return false;
        } break;
    case GGML_OP_DUP:
        {
            ggml_type src0_type = op->src[0]->type;
            return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
        } break;
    case GGML_OP_ADD:
    case GGML_OP_RMS_NORM:
    case GGML_OP_MUL:
    case GGML_OP_CONT:
    case GGML_OP_SOFT_MAX:
        return true;
    case GGML_OP_ROPE:
        return ggml_is_contiguous(op->src[0]);
    default:
        return false;
    }
    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_ascend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // todo if split: marked undo


    if (ggml_backend_buft_is_ascend(buft)) {
        auto ctx = (ggml_backend_ascend_context *)backend->context;
        auto buft_ctx = (ggml_backend_ascend_buffer_type_context *)buft->context;
        return buft_ctx->device == ctx->device;
    }
    return false;
}

GGML_CALL static bool ggml_backend_ascend_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(backend);
}

// todo event is optional: marked
static ggml_backend_event_t ggml_backend_ascend_event_new(ggml_backend_t backend) {
    // todo no peer copy: marked
    return nullptr;
}

static void ggml_backend_ascend_event_free(ggml_backend_event_t event) {
    // todo check: fixed
    auto ret = aclrtDestroyEvent(event->context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl destroy event failed at [ggml_backend_ascend_event_free]: %d\n", ret));
    delete event;
}

static void ggml_backend_ascend_event_record(ggml_backend_event_t event) {
    auto ctx = (ggml_backend_ascend_context *)event->backend->context;

    //todo check
    auto ret = aclrtRecordEvent(event->context, ctx->stream());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl record event failed at [ggml_backend_ascend_event_record]: %d\n", ret));
}

static void ggml_backend_ascend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto ctx = (ggml_backend_ascend_context *)backend->context;

    if (ggml_backend_is_ascend(event->backend)) {
        // todo Check
        auto ret = aclrtStreamWaitEvent(ctx->stream(), (aclrtEvent)event->context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl StreamWaitEvent failed at [ggml_backend_ascend_event_wait]: %d\n", ret); return);
    } else {
        GGML_ASSERT(false);
    }
}

static void ggml_backend_ascend_event_synchronize(ggml_backend_event_t event) {
    // todo Check
    auto ret = aclrtSynchronizeEvent((aclrtEvent)event->context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("acl SynchronizeEvent failed at [ggml_backend_ascend_event_synchronize]: %d\n", ret));
}

static ggml_backend_i ggml_backend_ascend_interface = {
    /* .get_name                  */ ggml_backend_ascend_name,
    /* .free                      */ ggml_backend_ascend_free,
    /* .get_default_buffer_type = */ ggml_backend_ascend_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_ascend_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_ascend_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_ascend_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_ascend_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ascend_graph_compute,
    /* .supports_op             = */ ggml_backend_ascend_supports_op,
    /* .supports_buft           = */ ggml_backend_ascend_supports_buft,
    /* .offload_op              = */ ggml_backend_ascend_offload_op,
    /* .event_new               = */ ggml_backend_ascend_event_new,
    /* .event_free              = */ ggml_backend_ascend_event_free,
    /* .event_record            = */ ggml_backend_ascend_event_record,
    /* .event_wait              = */ ggml_backend_ascend_event_wait,
    /* .event_synchronize       = */ ggml_backend_ascend_event_synchronize,
};

static ggml_guid_t ggml_backend_ascend_guid() {
    static ggml_guid guid = { 0xde, 0xad, 0xbe, 0xef, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, 0x15, 0x22, 0x37, 0x59, 0x90};
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_ascend_init(int device) {
    if (device < 0 || device >= ggml_backend_ascend_get_device_count()) {
        GGML_ASCEND_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    auto ctx = new ggml_backend_ascend_context(device);
    if (ctx == nullptr) {
        GGML_ASCEND_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    auto ascend_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_ascend_guid(),
        /* .interface = */ ggml_backend_ascend_interface,
        /* .context   = */ ctx
    };

    return ascend_backend;
}