#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#define GGML_ASCEND_NAME "ASCEND"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_ASCEND_MAX_DEVICES       16

GGML_API GGML_CALL ggml_backend_t ggml_backend_ascend_init(int device);

GGML_API GGML_CALL bool ggml_backend_is_ascend(ggml_backend_t backend);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_ascend_buffer_type(int device);

// GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_ascend_split_buffer_type(const float * tensor_split);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_ascend_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_ascend_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_ascend_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_ascend_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL bool ggml_backend_ascend_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_ascend_unregister_host_buffer(void * buffer);

GGML_API void ggml_backend_ascend_log_set_callback(ggml_log_callback log_callback, void * user_data);
#ifdef  __cplusplus
}
#endif