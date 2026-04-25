#include <cuda_runtime.h>
#include <cstdlib>
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

extern "C" {

// Memory allocation
void* cuda_alloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_alloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

// Memory free
void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// Host to device copy
void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

// Device to host copy
void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

// Device synchronization
void cuda_sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Memory set
void cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

// Allocate and copy host to device
void* cuda_alloc_and_copy(const void* host_ptr, size_t size) {
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_alloc_and_copy failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));
    return dev_ptr;
}

} // extern "C"