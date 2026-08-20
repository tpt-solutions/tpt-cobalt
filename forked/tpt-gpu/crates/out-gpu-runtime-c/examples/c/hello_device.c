/*
 * hello_device.c - Minimal example using the TPT Runtime C ABI.
 *
 * Build (after `cargo build -p out-gpu-runtime-c`):
 *   cc examples/c/hello_device.c \
 *      -L ../../target/debug -ltptr_c -o hello_device
 *   (or -L ../../target/release -ltptr_c for the release build)
 *
 * On Windows, link tptr_c.dll; on macOS, link libtptr_c.dylib.
 */
#include "tptr/tptr_capi.h"
#include <stdio.h>
#include <string.h>

#define TOTAL_MEM (16ull * 1024 * 1024 * 1024)

int main(void) {
    tptr_device_t *dev = NULL;
    if (tptr_device_create(0, TOTAL_MEM, &dev) != TptrOk) {
        fprintf(stderr, "failed to create device: %s\n", tptr_last_error());
        return 1;
    }

    printf("TPT Runtime C ABI v%u.%u.%u\n",
           tptr_version_major(), tptr_version_minor(), tptr_version_patch());

    tptr_memory_t *mem = NULL;
    if (tptr_device_allocate(dev, 16, 0 /* device */, 0 /* read_write */, &mem) != TptrOk) {
        fprintf(stderr, "allocate failed: %s\n", tptr_last_error());
        tptr_device_destroy(dev);
        return 1;
    }

    const uint8_t host_src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    if (tptr_device_memcpy_htod(dev, mem, host_src, 16, 0) != TptrOk) {
        fprintf(stderr, "memcpy_htod failed: %s\n", tptr_last_error());
        tptr_device_free(dev, mem);
        tptr_device_destroy(dev);
        return 1;
    }

    uint8_t host_dst[16] = {0};
    if (tptr_device_memcpy_dtoh(dev, mem, host_dst, 16, 0) != TptrOk) {
        fprintf(stderr, "memcpy_dtoh failed: %s\n", tptr_last_error());
        tptr_device_free(dev, mem);
        tptr_device_destroy(dev);
        return 1;
    }

    if (memcmp(host_src, host_dst, 16) == 0) {
        printf("memcpy round-trip OK (byte-for-byte equal)\n");
    } else {
        fprintf(stderr, "memcpy round-trip MISMATCH\n");
    }

    tptr_kernel_t *kernel = NULL;
    if (tptr_device_create_kernel(dev, "hello", &kernel) == TptrOk) {
        tptr_kernel_config_t *cfg = NULL;
        if (tptr_kernel_config_create(1, 1, 1, 256, 1, 1, &cfg) == TptrOk) {
            tptr_kernel_handle_t *h = NULL;
            if (tptr_device_launch_kernel(dev, kernel, cfg, &h) == TptrOk) {
                printf("kernel launch handle complete: %s\n",
                       tptr_kernel_handle_is_complete(h) ? "yes" : "no");
                tptr_kernel_handle_destroy(h);
            }
            tptr_kernel_config_destroy(cfg);
        }
        tptr_kernel_destroy(kernel);
    }

    tptr_device_synchronize(dev);
    tptr_device_free(dev, mem);

    /* --- load_module path: compile a hand-written TPTIR module --- */
    {
        const char *module =
            "module {\n"
            "  func.func @reduce_max(%in: memref<*xf32>, %out: memref<*xf32>) attributes {tptir.kernel} {\n"
            "    ^entry:\n"
            "      %v = tptir.load(%in)\n"
            "      %m = tptir.max(%v)\n"
            "      tptir.store(%m, %out)\n"
            "      tptir.return\n"
            "  }\n"
            "}\n";

        tptr_kernel_t *k = NULL;
        if (tptr_device_load_module(dev, module, &k) == TptrOk) {
            tptr_kernel_config_t *cfg = NULL;
            if (tptr_kernel_config_create(1, 1, 1, 1, 1, 1, &cfg) == TptrOk) {
                tptr_kernel_handle_t *h = NULL;
                if (tptr_device_launch_kernel(dev, k, cfg, &h) == TptrOk) {
                    printf("load_module kernel launch complete: %s\n",
                           tptr_kernel_handle_is_complete(h) ? "yes" : "no");
                    tptr_kernel_handle_destroy(h);
                }
                tptr_kernel_config_destroy(cfg);
            }
            tptr_kernel_destroy(k);
            printf("load_module: TPTIR module compiled and launched OK\n");
        } else {
            fprintf(stderr, "load_module failed: %s\n", tptr_last_error());
        }
    }

    tptr_device_destroy(dev);
    printf("done\n");
    return 0;
}
