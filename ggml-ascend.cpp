#include "ggml.h"
#include "ggml-ascend.h"
#include "ggml-backend-impl.h"

// Fixme: This file including is not suitable
#include "acl/acl.h"

#include <algorithm>
#include <array>
#include <atomic>
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

void ggml_ascend_set_device(int device) {
    int current_device;
    // todo Check
    aclrtGetDevice(&current_device);

    if (device == current_device) {
        return;
    }

    // todo Check
    aclrtSetDevice(device);
}

int ggml_ascend_get_device() {
    int id;
    // todo check
    aclrtGetDevice(&id);
    return id;
}

static aclError ggml_ascend_device_malloc(void ** ptr, size_t size, int device) {
    ggml_ascend_set_device(device);

    return aclrtMalloc(ptr, size);
}

struct ggml_backend_ascend_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_ascend_buffer_context() :
        device(device), dev_ptr(dev_ptr), 
        name(GGML_ASCEND_NAME + std::to_string(device)) {
    } 

    ~ggml_backend_ascend_buffer_context() {
        // todo Check
        aclrtFree(dev_ptr);
    }
} 

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
            // todo Check
            aclrtMemset(static_cast<char *>(tensor->data) + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_ascend_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto ctx = (ggml_backend_ascend_buffer_context *)buffer->context;

    ggml_ascend_set_device(ctx->device);

    // todo Check
    aclrtMemcpy((char *)tensor->data, offset, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

GGML_CALL static void ggml_backend_ascend_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto ctx = (ggml_backend_ascend_buffer_context *)buffer->context;

    ggml_ascend_set_device(ctx->device);
    //todo Check
    aclrtMemcpy(data, (const char *)tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST);
}

GGML_CALL static bool ggml_backend_ascend_buffer_cpy_tensor(ggml_bcakend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_ascend(src->buffer)) {
        auto src_ctx = (ggml_backend_ascend_buffer_context *)src->buffer->context;
        auto dst_ctx = (ggml_backend_ascend_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            // todo Check
            // Now we don't support peer copy
            aclrtMemcpy(dst->data, ggml_nbytes(src), src->data, ggml_nbytes(src), ACL_MEMCPY_DEVICE_TO_DEVICE);
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
    // todo Check
    aclrtSynchronizeDevice();
    aclrtMemset(ctx->dev_ptr, value, buffer->size);
    aclrtSynchronizeDevice();
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
}

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
        // todo log
        return nullptr;
    }

    auto ctx = new ggml_backend_ascend_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_ascend_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_ascend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128; // todo REALLY?

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

                // todo implement！！！！！！！！！
            };
        }
        ggml_backend_ascend_buffer_type = true;
    }

    return &ggml_backend_ascend_buffer_types[device];
}

GGML_CALL int ggml_backend_ascend_get_device_count() {
    uint32_t ret = 0;
    auto err = aclrtGetDeviceCount(&ret);
    return err == 0 ? ret : 0;
}

GGML_CALL static const char * ggml_backend_ascend_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_ASCEND_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_ascend_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_ASCEND_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_ascend_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    aclrtFreeHost(buffer->context);
    //todo ACL_CHECK
}

static void *  ggml_ascend_host_malloc(size_t size) {
    if (getenv("GGML_ASCEND_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    aclError err = aclrtMallocHost((void **) &ptr, size);
    if (err != ACL_SUCCESS) {
        //todo: ERROR LOG
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

GGML_CALL void ggml_backend_ascend_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_ascend_set_device(device);

    // todo Check
    aclrtGetMemInfo(ACL_DDR_MEM, free, total);
}