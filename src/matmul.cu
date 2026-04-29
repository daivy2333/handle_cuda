#include "cuda_ops.h"
#include "cuda_util.h"
#include <cmath>

namespace {

#define TILE_SIZE 32

// ============================================================================
// Optimized matmul kernels with shared memory tiling
// ============================================================================

// Standard tiled matmul: C = A @ B
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A [M, K]
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B [K, N]
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled matmul with A transposed: C = A^T @ B
// A is [K, M] (stored), we access as [M, K] logically
// B is [K, N]
// C is [M, N]
__global__ void matmul_tiled_transA_kernel(const float* A, const float* B, float* C,
                                            int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // Will hold A^T tile
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // row in C (row in A^T)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // col in C (col in B)

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A^T (which is A stored as [K, M])
        // We want A^T[row, t*TILE_SIZE + threadIdx.x] = A[t*TILE_SIZE + threadIdx.x, row]
        int a_row = t * TILE_SIZE + threadIdx.x;  // row in stored A
        int a_col = row;                          // col in stored A
        if (a_row < K && a_col < M) {
            As[threadIdx.y][threadIdx.x] = A[a_row * M + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B [K, N]
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled matmul with B transposed: C = A @ B^T
// A is [M, K]
// B is [N, K] (stored), we access as [K, N] logically
// C is [M, N]
__global__ void matmul_tiled_transB_kernel(const float* A, const float* B, float* C,
                                            int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];  // Will hold B^T tile

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A [M, K]
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B^T (which is B stored as [N, K])
        // We want B^T[t*TILE_SIZE + threadIdx.y, col] = B[col, t*TILE_SIZE + threadIdx.y]
        int b_row = col;                          // row in stored B
        int b_col = t * TILE_SIZE + threadIdx.y;  // col in stored B
        if (b_row < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled matmul with both transposed: C = A^T @ B^T
// A is [K, M] (stored)
// B is [N, K] (stored)
// C is [M, N]
__global__ void matmul_tiled_transAB_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A^T: A[t*TILE_SIZE + threadIdx.x, row]
        int a_row = t * TILE_SIZE + threadIdx.x;
        int a_col = row;
        if (a_row < K && a_col < M) {
            As[threadIdx.y][threadIdx.x] = A[a_row * M + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B^T: B[col, t*TILE_SIZE + threadIdx.y]
        int b_row = col;
        int b_col = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Backward kernels (also can be optimized)
// ============================================================================

// Naive backward kernels (small matrices usually, so acceptable)
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

// Naive general matmul (fallback)
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

__global__ void matmul_transpose_kernel_naive(const float* A, const float* B, float* C,
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

// ============================================================================
// Main matmul function with optimized transpose kernels
// ============================================================================

void cuda_matmul(const float* A, const float* B, float* C, const MatMulDesc& desc, cudaStream_t stream) {
    int M = static_cast<int>(desc.M);
    int N = static_cast<int>(desc.N);
    int K = static_cast<int>(desc.K);

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    if (!desc.transpose_a && !desc.transpose_b) {
        // C = A @ B (standard)
        matmul_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    } else if (desc.transpose_a && !desc.transpose_b) {
        // C = A^T @ B (optimized)
        // A is stored as [K, M], we compute A^T @ B = [M, K] @ [K, N] = [M, N]
        matmul_tiled_transA_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    } else if (!desc.transpose_a && desc.transpose_b) {
        // C = A @ B^T (optimized)
        // B is stored as [N, K], we compute A @ B^T = [M, K] @ [K, N] = [M, N]
        matmul_tiled_transB_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    } else {
        // C = A^T @ B^T (optimized)
        matmul_tiled_transAB_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
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