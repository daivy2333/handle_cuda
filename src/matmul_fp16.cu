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
