// src/matmul_cublas.cu
#include "cuda_ops.h"
#include "cuda_util.h"
#include <cublas_v2.h>

void cuda_matmul_cublas(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K, cudaStream_t stream) {
    // Static handle: lives for process lifetime, never destroyed intentionally
    static cublasHandle_t handle = []{
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();

    if (stream) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
    }

    float alpha = 1.0f, beta = 0.0f;
    // cublasSgemm computes C = alpha * op(B) * op(A) + beta * C
    // Dimensions: (N, M, K) = (output cols, output rows, inner dimension)
    // B is (N x K), A is (K x M), C is (N x M)
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N));
}