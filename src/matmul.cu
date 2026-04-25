#include "cuda_ops.h"
#include "cuda_util.h"
#include <cublas_v2.h>

namespace {

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K,
                               int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * lda + i] * B[i * ldb + col];
        }
        C[row * ldc + col] = sum;
    }
}

__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K,
                                         bool trans_a, bool trans_b,
                                         int lda, int ldb, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            int a_row = trans_a ? i : row;
            int a_col = trans_a ? row : i;
            int b_row = trans_b ? col : i;
            int b_col = trans_b ? i : col;
            sum += A[a_row * lda + a_col] * B[b_row * ldb + b_col];
        }
        C[row * ldc + col] = sum;
    }
}

__global__ void matmul_backward_A_kernel(const float* grad_C, const float* B,
                                         float* grad_A, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += grad_C[row * N + i] * B[col * N + i];
        }
        grad_A[row * K + col] = sum;
    }
}

__global__ void matmul_backward_B_kernel(const float* grad_C, const float* A,
                                         float* grad_B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[i * K + row] * grad_C[i * N + col];
        }
        grad_B[row * N + col] = sum;
    }
}

} // namespace

void cuda_matmul(const float* A, const float* B, float* C, const MatMulDesc& desc, cudaStream_t stream) {
    int M = static_cast<int>(desc.M);
    int N = static_cast<int>(desc.N);
    int K = static_cast<int>(desc.K);

    int lda = desc.transpose_a ? M : K;
    int ldb = desc.transpose_b ? K : N;
    int ldc = N;

    dim3 block_dim(16, 16);
    dim3 grid_dim((N + 15) / 16, (M + 15) / 16);

    if (desc.transpose_a || desc.transpose_b) {
        matmul_transpose_kernel<<<grid_dim, block_dim, 0, stream>>>(
            A, B, C, M, N, K, desc.transpose_a, desc.transpose_b, lda, ldb, ldc);
    } else {
        matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(
            A, B, C, M, N, K, lda, ldb, ldc);
    }

    CUDA_CHECK(cudaGetLastError());
}

void cuda_matmul_backward(const float* grad_C, const float* A, const float* B,
                           float* grad_A, float* grad_B,
                           const MatMulDesc& desc, cudaStream_t stream) {
    int M = static_cast<int>(desc.M);
    int N = static_cast<int>(desc.N);
    int K = static_cast<int>(desc.K);

    dim3 block_dim(16, 16);

    // grad_A = grad_C @ B^T (M x K)
    dim3 grid_A((K + 15) / 16, (M + 15) / 16);
    matmul_backward_A_kernel<<<grid_A, block_dim, 0, stream>>>(
        grad_C, B, grad_A, M, N, K);

    // grad_B = A^T @ grad_C (K x N)
    dim3 grid_B((N + 15) / 16, (K + 15) / 16);
    matmul_backward_B_kernel<<<grid_B, block_dim, 0, stream>>>(
        grad_C, A, grad_B, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}
