#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace {

// Float to Half conversion kernel (vectorized)
__global__ void float_to_half_kernel(const float* in, __half* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// Half to Float conversion kernel (vectorized)
__global__ void half_to_float_kernel(const __half* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// Vectorized float4 to half2 conversion
__global__ void float4_to_half2_kernel(const float4* in, half2* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = in[idx];
        out[idx] = __float22half2_rn(make_float2(val.x, val.y));
    }
}

// Vectorized half2 to float4 conversion
__global__ void half2_to_float4_kernel(const half2* in, float4* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half2 val = in[idx];
        float2 f = __half22float2(val);
        out[idx] = make_float4(f.x, f.y, 0.0f, 0.0f);
    }
}

} // namespace

// Host端转换函数
void float_to_half(const float* in, __half* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    float_to_half_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}

void half_to_float(const __half* in, float* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    half_to_float_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}

// Vectorized versions for better performance
void float4_to_half2(const float4* in, half2* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    float4_to_half2_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}

void half2_to_float4(const half2* in, float4* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    half2_to_float4_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}
