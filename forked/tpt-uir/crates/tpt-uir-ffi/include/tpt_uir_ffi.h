#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Deserialize a postcard-encoded region from `data[0..len]` into a new handle.
 *
 * On success, `*out_handle` receives the opaque pointer (free with
 * [`tpt_uir_free`]). Returns `0` on success, `1` on a bad pointer, or `2` on a
 * decode error.
 *
 * # Safety
 * `data` must point to `len` readable bytes.
 */
int tpt_uir_load(const uint8_t *data, size_t len, void **out_handle);

/**
 * Free a handle produced by [`tpt_uir_load`].
 *
 * # Safety
 * `handle` must be a pointer returned by [`tpt_uir_load`] and not previously
 * freed. Passing a null pointer is a no-op.
 */
void tpt_uir_free(void *handle);

/**
 * Write the number of blocks in the region to `*out`.
 *
 * # Safety
 * `handle` must be a valid handle; `out` must be non-null.
 */
int tpt_uir_block_count(void *handle, size_t *out);

/**
 * Write the number of operations in block `block_index` to `*out`.
 *
 * # Safety
 * `handle` must be a valid handle; `out` must be non-null.
 */
int tpt_uir_op_count(void *handle, size_t block_index, size_t *out);

/**
 * Write the dialect portion of operation `(block_index, op_index)` to `*out`.
 *
 * The returned pointer references storage owned by the handle and is valid
 * until [`tpt_uir_free`].
 *
 * # Safety
 * `handle` must be a valid handle; `out` must be non-null.
 */
int tpt_uir_op_dialect(void *handle, size_t block_index, size_t op_index, const char **out);

/**
 * Write the op portion of operation `(block_index, op_index)` to `*out`.
 *
 * See [`tpt_uir_op_dialect`] for lifetime semantics.
 *
 * # Safety
 * `handle` must be a valid handle; `out` must be non-null.
 */
int tpt_uir_op_op(void *handle, size_t block_index, size_t op_index, const char **out);

/**
 * Write the number of operands of operation `(block_index, op_index)` to `*out`.
 *
 * # Safety
 * `handle` must be a valid handle; `out` must be non-null.
 */
int tpt_uir_op_operand_count(void *handle, size_t block_index, size_t op_index, size_t *out);

/**
 * Write operand `operand_index` of operation `(block_index, op_index)` to `*out`.
 *
 * # Safety
 * `handle` must be a valid handle; `out` must be non-null.
 */
int tpt_uir_op_operand(void *handle,
                       size_t block_index,
                       size_t op_index,
                       size_t operand_index,
                       uint32_t *out);

/**
 * Validate SSA well-formedness. Returns `0` if valid, `1` if invalid.
 *
 * # Safety
 * `handle` must be a valid handle.
 */
int tpt_uir_validate_ssa(void *handle);

/**
 * Validate that every op's dialect starts with `prefix`. Returns `0` if all
 * match, `1` otherwise (or on a bad `prefix` pointer).
 *
 * # Safety
 * `handle` must be a valid handle; `prefix` must be a valid NUL-terminated C string.
 */
int tpt_uir_validate_dialect(void *handle, const char *prefix);
