#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

struct CudaBuffer {
    float* data = nullptr;
    size_t size = 0;

    CudaBuffer() = default;

    explicit CudaBuffer(size_t size) : size(size) {
        CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
    }

    ~CudaBuffer() {
        if (data) cudaFree(data);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void allocate(size_t sz) {
        if (data) cudaFree(data);
        size = sz;
        CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
    }

    void clear() {
        if (data) CUDA_CHECK(cudaMemset(data, 0, size * sizeof(float)));
    }
};

inline int get_num_blocks(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}

template<typename T>
T* host_to_device(const T* host_data, size_t count) {
    T* dev_data;
    CUDA_CHECK(cudaMalloc(&dev_data, count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    return dev_data;
}

template<typename T>
void host_to_device_async(T* dev_data, const T* host_data, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpy(dev_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void device_to_host(const T* dev_data, T* host_data, size_t count) {
    CUDA_CHECK(cudaMemcpy(host_data, dev_data, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void device_to_device(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

// Vectorized load/store helpers for float4 operations
__device__ __forceinline__ void load_float4(const float* ptr, float& a, float& b, float& c, float& d) {
    float4 val = *reinterpret_cast<const float4*>(ptr);
    a = val.x; b = val.y; c = val.z; d = val.w;
}

__device__ __forceinline__ void store_float4(float* ptr, float a, float b, float c, float d) {
    float4 val = make_float4(a, b, c, d);
    *reinterpret_cast<float4*>(ptr) = val;
}
