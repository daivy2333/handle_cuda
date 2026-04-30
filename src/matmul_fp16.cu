#include <cuda_fp16.h>
#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

// FP16 tiled matmul without Tensor Core (works on all GPUs)
// A and B are FP16, C is FP32 accumulator
__global__ void matmul_fp16_tiled_kernel(
    const __half* A, const __half* B, float* C,
    int M, int N, int K) {

    __shared__ __half As[16][16];
    __shared__ __half Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Load A tile
        if (row < M && t * 16 + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = __half(0);
        }

        // Load B tile
        if (col < N && t * 16 + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = __half(0);
        }

        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// FP32 Matmul baseline using existing tiled implementation
__global__ void matmul_fp32_tiled_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + 31) / 32; ++t) {
        // Load A tile
        if (row < M && t * 32 + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        if (col < N && t * 32 + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tensor Core Matmul kernel (requires compute capability 7.0+)
#if __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Optimized Tensor Core kernel: multiple warps per block + shared memory staging
// Each block computes 64x64 output region (4 warps x 2 warps = 8 tiles of 16x16)
__global__ void tensor_core_matmul_optimized_kernel(
    const __half* A, const __half* B, float* C,
    int M, int N, int K) {

    // Block computes 64x64 output region (8 tiles)
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int WARP_COUNT = 8;  // 4x2 layout

    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    // Warp ID: 0-7, arranged as 4 columns x 2 rows
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int warp_row_idx = warp_id / 4;   // 0 or 1
    int warp_col_idx = warp_id % 4;   // 0, 1, 2, or 3

    // Each warp computes 16x16 output tile
    int warp_row = block_row + warp_row_idx * WMMA_M;
    int warp_col = block_col + warp_col_idx * WMMA_N;

    // Shared memory for cooperative loading
    // Layout: [BLOCK_M][WMMA_K] for A, [WMMA_K][BLOCK_N] for B
    __shared__ __half As[BLOCK_M][WMMA_K];
    __shared__ __half Bs[WMMA_K][BLOCK_N];

    // Accumulator
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
        // Cooperative load of A and B tiles
        // 256 threads load 64*16 = 1024 elements for A, and 16*64 = 1024 for B
        // Each thread loads 4 elements
        for (int i = 0; i < 4; ++i) {
            int tid = threadIdx.x + i * 256;

            // Load A: [BLOCK_M][WMMA_K] = [64][16]
            int a_row = tid / WMMA_K;
            int a_col = tid % WMMA_K;
            int global_a_row = block_row + a_row;
            int global_a_col = k_tile + a_col;
            if (global_a_row < M && global_a_col < K) {
                As[a_row][a_col] = A[global_a_row * K + global_a_col];
            } else {
                As[a_row][a_col] = __half(0);
            }

            // Load B: [WMMA_K][BLOCK_N] = [16][64]
            int b_row = tid / BLOCK_N;
            int b_col = tid % BLOCK_N;
            int global_b_row = k_tile + b_row;
            int global_b_col = block_col + b_col;
            if (global_b_row < K && global_b_col < N) {
                Bs[b_row][b_col] = B[global_b_row * N + global_b_col];
            } else {
                Bs[b_row][b_col] = __half(0);
            }
        }

        __syncthreads();

        // Each warp loads its portion and computes
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;

        // Load A[warp_row:16, :] with stride WMMA_K
        load_matrix_sync(a_frag, &As[warp_row_idx * WMMA_M][0], WMMA_K);
        // Load B[:, warp_col:16] with stride BLOCK_N
        load_matrix_sync(b_frag, &Bs[0][warp_col_idx * WMMA_N], BLOCK_N);

        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store result
    if (warp_row < M && warp_col < N) {
        store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, row_major);
    }
}

// Basic Tensor Core kernel (for reference)
__global__ void tensor_core_matmul_kernel(
    const __half* A, const __half* B, float* C,
    int M, int N, int K) {

    int row_o = blockIdx.y * WMMA_M;
    int col_o = blockIdx.x * WMMA_N;

    if (row_o >= M || col_o >= N) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = row_o;
        int a_col = k;
        if (a_row < M && a_col < K) {
            load_matrix_sync(a_frag, A + a_row * K + a_col, K);
        } else {
            fill_fragment(a_frag, __half(0));
        }

        int b_row = k;
        int b_col = col_o;
        if (b_row < K && b_col < N) {
            load_matrix_sync(b_frag, B + b_row * N + b_col, N);
        } else {
            fill_fragment(b_frag, __half(0));
        }

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + row_o * N + col_o, c_frag, N, row_major);
}
#endif

} // namespace

// Public API
void cuda_matmul_fp16(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream) {

    dim3 grid_dim((N + 15) / 16, (M + 15) / 16);
    dim3 block_dim(16, 16);

    matmul_fp16_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// Tensor Core optimized API (requires compute capability 7.0+)
void cuda_matmul_fp16_tensor_core(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream) {

    #if __CUDA_ARCH__ >= 700
    // Use optimized kernel: 64x64 output per block, 256 threads
    dim3 grid_dim((N + 63) / 64, (M + 63) / 64);
    dim3 block_dim(256);  // 8 warps per block

    tensor_core_matmul_optimized_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    #else
    // Fallback to tiled kernel if Tensor Core not available
    cuda_matmul_fp16(A, B, C, M, N, K, stream);
    #endif
}

void cuda_matmul_fp32_baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream) {

    dim3 grid_dim((N + 31) / 32, (M + 31) / 32);
    dim3 block_dim(32, 32);

    matmul_fp32_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_matmul_fp16_naive(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream) {

    dim3 grid_dim((N + 15) / 16, (M + 15) / 16);
    dim3 block_dim(16, 16);

    matmul_fp16_tiled_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// FP16 Matmul Backward kernels
// grad_C: [M, N], A: [M, K], B: [K, N]
// grad_A: [M, K] = grad_C @ B^T
// grad_B: [K, N] = A^T @ grad_C

namespace {

// Tiled FP16 backward kernels - corrected index mapping for precision
// grad_B = A^T @ grad_C: [K, N]
// A: [M, K], grad_C: [M, N], grad_B: [K, N]
// grad_B[row][col] = sum over m of A[m][row] * grad_C[m][col]
__global__ void matmul_fp16_backward_B_tiled_kernel(
    const __half* A, const float* grad_C, float* grad_B,
    int M, int K, int N) {

    int row = blockIdx.y * 16 + threadIdx.y;  // K dimension
    int col = blockIdx.x * 16 + threadIdx.x;  // N dimension

    __shared__ __half As[16][16];   // A[m_local][k_local]
    __shared__ float Cs[16][16];    // grad_C[m_local][n_local]

    float sum = 0.0f;

    for (int t = 0; t < (M + 15) / 16; ++t) {
        int m_idx = t * 16 + threadIdx.y;

        // Load A[m_idx][blockIdx.y*16+threadIdx.x] - row varies by threadIdx.x
        int k_idx = blockIdx.y * 16 + threadIdx.x;
        if (m_idx < M && k_idx < K) {
            As[threadIdx.y][threadIdx.x] = A[m_idx * K + k_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = __half(0);
        }

        // Load grad_C[m_idx][blockIdx.x*16+threadIdx.x]
        int n_idx = blockIdx.x * 16 + threadIdx.x;
        if (m_idx < M && n_idx < N) {
            Cs[threadIdx.y][threadIdx.x] = grad_C[m_idx * N + n_idx];
        } else {
            Cs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate: A[t*16+m_local][row] * grad_C[t*16+m_local][col]
        // As[m_local][threadIdx.y] = A[t*16+m_local][blockIdx.y*16+threadIdx.y] = A[t*16+m_local][row]
        // Cs[m_local][threadIdx.x] = grad_C[t*16+m_local][blockIdx.x*16+threadIdx.x] = grad_C[t*16+m_local][col]
        for (int m_local = 0; m_local < 16; ++m_local) {
            sum += __half2float(As[m_local][threadIdx.y]) * Cs[m_local][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        grad_B[row * N + col] = sum;
    }
}

// grad_A = grad_C @ B^T: [M, K]
// grad_C: [M, N], B: [K, N], grad_A: [M, K]
// grad_A[row][col] = sum over n of grad_C[row][n] * B[col][n]
__global__ void matmul_fp16_backward_A_tiled_kernel(
    const float* grad_C, const __half* B, float* grad_A,
    int M, int N, int K) {

    int row = blockIdx.y * 16 + threadIdx.y;  // M dimension
    int col = blockIdx.x * 16 + threadIdx.x;  // K dimension

    __shared__ float Cs[16][16];    // grad_C[m_local][n_local]
    __shared__ __half Bs[16][16];   // B[k_local][n_local]

    float sum = 0.0f;

    for (int t = 0; t < (N + 15) / 16; ++t) {
        int n_idx = t * 16 + threadIdx.x;

        // Load grad_C[blockIdx.y*16+threadIdx.y][t*16+threadIdx.x]
        int m_idx = blockIdx.y * 16 + threadIdx.y;
        if (m_idx < M && n_idx < N) {
            Cs[threadIdx.y][threadIdx.x] = grad_C[m_idx * N + n_idx];
        } else {
            Cs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B[blockIdx.x*16+threadIdx.x][t*16+threadIdx.y]
        int k_idx = blockIdx.x * 16 + threadIdx.x;
        int n_idx2 = t * 16 + threadIdx.y;
        if (k_idx < K && n_idx2 < N) {
            Bs[threadIdx.x][threadIdx.y] = B[k_idx * N + n_idx2];
        } else {
            Bs[threadIdx.x][threadIdx.y] = __half(0);
        }

        __syncthreads();

        // Accumulate: grad_C[row][t*16+n] * B[col][t*16+n]
        // Cs[threadIdx.y][n_local] = grad_C[blockIdx.y*16+threadIdx.y][t*16+n_local] = grad_C[row][t*16+n_local]
        // Bs[threadIdx.x][n_local] = B[blockIdx.x*16+threadIdx.x][t*16+n_local] = B[col][t*16+n_local]
        for (int n_local = 0; n_local < 16; ++n_local) {
            sum += Cs[threadIdx.y][n_local] * __half2float(Bs[threadIdx.x][n_local]);
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        grad_A[row * K + col] = sum;
    }
}

// Naive FP16 backward kernels - fallback for small matrices
__global__ void matmul_fp16_backward_B_naive_kernel(
    const __half* A, const float* grad_C, float* grad_B,
    int M, int K, int N) {

    int row = blockIdx.y * 16 + threadIdx.y;  // K dimension
    int col = blockIdx.x * 16 + threadIdx.x;  // N dimension

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            // A^T[row][i] = A[i][row] = A[i * K + row]
            sum += __half2float(A[i * K + row]) * grad_C[i * N + col];
        }
        grad_B[row * N + col] = sum;
    }
}

// grad_A = grad_C @ B^T: [M, K]
// grad_C: [M, N], B: [K, N], grad_A: [M, K]
__global__ void matmul_fp16_backward_A_naive_kernel(
    const float* grad_C, const __half* B, float* grad_A,
    int M, int N, int K) {

    int row = blockIdx.y * 16 + threadIdx.y;  // M dimension
    int col = blockIdx.x * 16 + threadIdx.x;  // K dimension

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            // B^T[col][i] = B[i][col] = B[col * N + i]... wait
            // B: [K, N], B^T: [N, K]
            // B^T[i][col] = B[col][i] = B[col * N + i]
            sum += grad_C[row * N + i] * __half2float(B[col * N + i]);
        }
        grad_A[row * K + col] = sum;
    }
}

} // namespace

void cuda_matmul_fp16_backward(
    const float* grad_C, const __half* A, const __half* B,
    float* grad_A, float* grad_B,
    int M, int N, int K, cudaStream_t stream) {

    // Use tiled kernels for better performance
    // grad_B = A^T @ grad_C: [K, N]
    if (grad_B) {
        dim3 grid_B((N + 15) / 16, (K + 15) / 16);
        dim3 block_B(16, 16);
        matmul_fp16_backward_B_tiled_kernel<<<grid_B, block_B, 0, stream>>>(
            A, grad_C, grad_B, M, K, N);
    }

    // grad_A = grad_C @ B^T: [M, K]
    if (grad_A) {
        dim3 grid_A((K + 15) / 16, (M + 15) / 16);
        dim3 block_A(16, 16);
        matmul_fp16_backward_A_tiled_kernel<<<grid_A, block_A, 0, stream>>>(
            grad_C, B, grad_A, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
}
