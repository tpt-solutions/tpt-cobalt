// Kernels for the tpt-runtime CUDA backend (compiled to PTX at build time;
// the .ptx artifact is embedded in cuda_backend.rs and JIT-compiled by the
// driver, so no nvcc is needed to RUN this crate — only to regenerate it).
extern "C" __global__ void add_f32(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void matmul_f32(const float* a, const float* b, float* c,
                                      int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; p++) acc += a[row * k + p] * b[p * n + col];
        c[row * n + col] = acc;
    }
}
