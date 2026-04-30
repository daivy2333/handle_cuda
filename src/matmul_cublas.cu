// src/matmul_cublas.cu
#include "cuda_ops.h"
#include <cublas_v2.h>

void cuda_matmul_cublas(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K, cudaStream_t stream) {
    static cublasHandle_t handle = []{
        cublasHandle_t h;
        cublasCreate(&h);
        return h;
    }();

    if (stream) {
        cublasSetStream(handle, stream);
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N);
}