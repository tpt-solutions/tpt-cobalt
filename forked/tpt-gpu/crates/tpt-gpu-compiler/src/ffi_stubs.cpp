// Stubs for the TPTIR C API symbols referenced from Rust FFI.
// These provide linkable definitions so the Rust crate can build without
// depending on the full C++ TPTIR/MLIR/LLVM toolchain. The default
// feature set now prefers the native Rust compiler path
// (`crate::compile_native`), and these stubs are only exercised when
// the `ffi` feature explicitly opts in.
#include <cstdlib>
#include <cstring>

struct tptir_string_t { char* data; size_t size; };
struct tptir_version_t { unsigned major, minor, patch; };

static int g_dummy = 0;

extern "C" {

int tptir_init(void** ctx) {
    if (!ctx) return -7;
    *ctx = &g_dummy;
    return 0;
}

int tptir_shutdown(void* ctx) {
    return 0;
}

tptir_version_t tptir_get_version() { return {0, 1, 0}; }

int tptir_compile(const char* source, size_t len, int /*target*/,
                  tptir_string_t* out, tptir_string_t* err) {
    if (!out) return -7;
    // Produce a **pass-through** textual representation of the source so
    // that `compile_via_ffi` returns *something* inspectable. Mirrors the
    // native Rust behaviour (`compile_native` -> `emit_tptir`) but skips
    // the pass pipeline, which is fine for the FFI path being a fallback.
    out->size = len;
    out->data = reinterpret_cast<char*>(std::malloc(len + 1));
    if (!out->data) return -1;
    std::memcpy(out->data, source, len);
    out->data[len] = '\0';
    if (err) { err->data = nullptr; err->size = 0; }
    return 0;
}

void tptir_string_free(tptir_string_t* s) {
    if (s && s->data) {
        std::free(s->data);
        s->data = nullptr;
        s->size = 0;
    }
}

} // extern "C"
