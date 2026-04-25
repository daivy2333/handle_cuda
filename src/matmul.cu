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
