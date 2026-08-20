#ifndef TPTIR_CAPI_TPTIR_CAPI_H
#define TPTIR_CAPI_TPTIR_CAPI_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  TPTIR_OK=0, TPTIR_ERROR_GENERIC=-1, TPTIR_ERROR_PARSE=-2,
  TPTIR_ERROR_TYPE=-3, TPTIR_ERROR_CODEGEN=-5, TPTIR_ERROR_NULL_POINTER=-7,
} tptir_status_t;
typedef void* tptir_context_t;
typedef void* tptir_module_t;
typedef void* tptir_region_t;
typedef void* tptir_block_t;
typedef void* tptir_operation_t;
typedef void* tptir_value_t;
typedef void* tptir_type_t;
typedef void* tptir_builder_t;
typedef void* tptir_pass_pipeline_t;
typedef void* tptir_codegen_t;
typedef void* tptir_parser_t;
typedef struct { char* data; size_t size; } tptir_string_t;
typedef struct { uint32_t major, minor, patch; } tptir_version_t;
tptir_status_t tptir_init(tptir_context_t* ctx);
tptir_status_t tptir_shutdown(tptir_context_t ctx);
tptir_version_t tptir_get_version(void);
const char* tptir_status_string(tptir_status_t status);
tptir_status_t tptir_module_create(tptir_context_t, tptir_module_t*);
tptir_status_t tptir_module_destroy(tptir_module_t);
tptir_status_t tptir_module_parse(tptir_module_t, const char*, size_t, tptir_string_t*);
tptir_status_t tptir_parser_create(tptir_context_t, const char*, size_t, tptir_parser_t*);
tptir_status_t tptir_parser_destroy(tptir_parser_t);
tptir_status_t tptir_parser_parse_function(tptir_parser_t, tptir_region_t*);
tptir_status_t tptir_builder_create(tptir_context_t, tptir_builder_t*);
tptir_status_t tptir_builder_destroy(tptir_builder_t);
tptir_status_t tptir_pass_pipeline_create(tptir_context_t, tptir_pass_pipeline_t*);
tptir_status_t tptir_pass_pipeline_destroy(tptir_pass_pipeline_t);
tptir_status_t tptir_pass_pipeline_add_pass(tptir_pass_pipeline_t, const char*);
tptir_status_t tptir_pass_pipeline_run(tptir_pass_pipeline_t, tptir_region_t, uint64_t*);
tptir_status_t tptir_codegen_create(tptir_context_t, int32_t, tptir_codegen_t*);
tptir_status_t tptir_codegen_destroy(tptir_codegen_t);
tptir_status_t tptir_codegen_generate(tptir_codegen_t, tptir_region_t, tptir_string_t*);
void tptir_string_free(tptir_string_t*);
tptir_status_t tptir_compile(const char*, size_t, int32_t, tptir_string_t*, tptir_string_t*);
#ifdef __cplusplus
}
#endif
#endif
