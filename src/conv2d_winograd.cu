#include "cuda_ops.h"
#include "cuda_util.h"
#include <cstdio>

namespace {

// Winograd F(2×2, 3×3) Transform Matrices (wincnn standard)
// A^T = [[1,1,1,0],[0,1,-1,1]], so A = [[1,0],[1,1],[1,-1],[0,1]]
__device__ __constant__ float winograd_A[4][4] = {
    { 1.0f,  0.0f,  0.0f,  0.0f},  // row 0: A[0] = [1, 0]
    { 1.0f,  1.0f,  0.0f,  0.0f},  // row 1: A[1] = [1, 1]
    { 1.0f, -1.0f,  0.0f,  0.0f},  // row 2: A[2] = [1, -1]
    { 0.0f,  1.0f,  0.0f,  0.0f}   // row 3: A[3] = [0, 1]
};

// B^T (input transform left side): B^T @ U @ B
__device__ __constant__ float winograd_Bt[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f},
    { 0.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f, -1.0f,  1.0f,  0.0f},
    { 0.0f,  1.0f,  0.0f, -1.0f}
};

// B (input transform right side): B^T @ U @ B
__device__ __constant__ float winograd_B[4][4] = {
    { 1.0f,  0.0f,  0.0f,  0.0f},
    { 0.0f,  1.0f, -1.0f,  1.0f},
    {-1.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f,  0.0f,  0.0f, -1.0f}
};

// G (weight transform): G @ w @ G^T
__device__ __constant__ float winograd_G[4][3] = {
    { 1.0f,  0.0f,  0.0f},
    { 0.5f,  0.5f,  0.5f},
    { 0.5f, -0.5f,  0.5f},
    { 0.0f,  0.0f,  1.0f}
};

// Weight Transform: [out_C, C, 3, 3] -> [out_C, C, 4, 4]
__global__ void winograd_weight_transform_kernel(
    const float* weight, float* weight_transformed,
    int out_C, int C) {
    
    int oc = blockIdx.x;
    int c = blockIdx.y;
    if (oc >= out_C || c >= C) return;
    
    float w[3][3];
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            w[kh][kw] = weight[oc * C * 9 + c * 9 + kh * 3 + kw];
        }
    }
    
    // W = G @ w @ G^T
    float temp[4][3] = {{0}};
    float W[4][4] = {{0}};
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                temp[i][j] += winograd_G[i][k] * w[k][j];
            }
        }
    }
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                W[i][j] += temp[i][k] * winograd_G[j][k];
            }
        }
    }
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            weight_transformed[oc * C * 16 + c * 16 + i * 4 + j] = W[i][j];
        }
    }
}

// Input Transform: V = B^T @ U @ B
__global__ void winograd_input_transform_kernel(
    const float* input, float* input_tiles,
    int N, int C, int H, int W,
    int num_tiles_h, int num_tiles_w,
    int out_H, int out_W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * num_tiles_h * num_tiles_w * 16;
    if (idx >= total) return;
    
    int tile_flat = idx / 16;
    int elem_idx = idx % 16;
    
    int n = tile_flat / (C * num_tiles_h * num_tiles_w);
    int rem = tile_flat % (C * num_tiles_h * num_tiles_w);
    int c = rem / (num_tiles_h * num_tiles_w);
    rem %= (num_tiles_h * num_tiles_w);
    int tile_h = rem / num_tiles_w;
    int tile_w = rem % num_tiles_w;
    
    // Get 4x4 input region
    float U[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int ih = tile_h * 2 + i - 1;
            int iw = tile_w * 2 + j - 1;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                U[i][j] = 0.0f;
            } else {
                U[i][j] = input[n * C * H * W + c * H * W + ih * W + iw];
            }
        }
    }
    
    // V = B^T @ U @ B
    float temp[4][4] = {{0}};
    float V[4][4] = {{0}};
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[i][j] += winograd_Bt[i][k] * U[k][j];
            }
        }
    }
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                V[i][j] += temp[i][k] * winograd_B[k][j];
            }
        }
    }

    input_tiles[tile_flat * 16 + elem_idx] = V[elem_idx / 4][elem_idx % 4];
}

// Element-wise multiplication
__global__ void winograd_elementwise_kernel(
    const float* input_tiles, const float* weight_tiles,
    float* intermediate,
    int N, int C, int out_C, int num_tiles) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * num_tiles * 16;
    if (idx >= total) return;
    
    int rem = idx;
    int n = rem / (out_C * num_tiles * 16);
    rem %= (out_C * num_tiles * 16);
    int oc = rem / (num_tiles * 16);
    rem %= (num_tiles * 16);
    int tile_idx = rem / 16;
    int elem_idx = rem % 16;
    
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = input_tiles[(n * C + c) * num_tiles * 16 + tile_idx * 16 + elem_idx];
        float w = weight_tiles[oc * C * 16 + c * 16 + elem_idx];
        sum += v * w;
    }

    intermediate[idx] = sum;
}

// Output Transform: Y = A^T @ M @ A
// For output position (oh, ow) within 2x2 tile:
// Y = sum_k=0^3 sum_l=0^3 A[ah][k] * M[k][l] * A[aw][l]
// where ah = oh + 1, aw = ow + 1 (1-based indexing for Winograd A matrix)
// Note: A[row][col] in our matrix, so A[ah][k] is row ah, col k
__global__ void winograd_output_transform_kernel(
    const float* intermediate, float* output,
    int N, int out_C, int out_H, int out_W,
    int num_tiles_h, int num_tiles_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;
    if (idx >= total) return;
    
    int rem = idx;
    int n = rem / (out_C * out_H * out_W);
    rem %= (out_C * out_H * out_W);
    int oc = rem / (out_H * out_W);
    rem %= (out_H * out_W);
    int oh = rem / out_W;
    int ow = rem % out_W;
    
    int tile_h = oh / 2;
    int tile_w = ow / 2;
    
    if (tile_h >= num_tiles_h || tile_w >= num_tiles_w) {
        output[idx] = 0.0f;
        return;
    }
    
    int pos_h = oh % 2;
    int pos_w = ow % 2;

    int tile_idx = tile_h * num_tiles_w + tile_w;
    float M[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            M[i][j] = intermediate[n * out_C * num_tiles_h * num_tiles_w * 16 +
                                   oc * num_tiles_h * num_tiles_w * 16 +
                                   tile_idx * 16 + i * 4 + j];
        }
    }

    // Y = A^T @ M @ A
    // Y[p,q] = sum_k sum_l A[k,p] * M[k,l] * A[l,q]
    // where p = pos_h, q = pos_w (0-based output positions within 2x2 tile)
    float Y = 0.0f;
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            Y += winograd_A[k][pos_h] * M[k][l] * winograd_A[l][pos_w];
        }
    }
    
    output[idx] = Y;
}

} // namespace

void cuda_conv2d_winograd_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_buffer,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream) {
    
    printf("[Winograd] Forward: N=%d, C=%d, H=%d, W=%d, out_C=%d\n", N, C, H, W, out_C);
    
    if (stride_h != 1 || stride_w != 1 || pad_h != 1 || pad_w != 1) {
        printf("[Winograd] ERROR: Only supports stride=1, pad=1\n");
        return;
    }
    
    int out_H = H - 2;
    int out_W = W - 2;
    int num_tiles_h = (out_H + 1) / 2;
    int num_tiles_w = (out_W + 1) / 2;
    int num_tiles = num_tiles_h * num_tiles_w;
    
    float* weight_transformed = temp_buffer;
    float* input_tiles = temp_buffer + out_C * C * 16;
    float* intermediate = input_tiles + N * C * num_tiles * 16;
    
    printf("[Winograd] out_H=%d, out_W=%d, tiles=%dx%d\n", out_H, out_W, num_tiles_h, num_tiles_w);
    
    dim3 wt_grid(out_C, C);
    winograd_weight_transform_kernel<<<wt_grid, 256, 0, stream>>>(
        weight, weight_transformed, out_C, C);
    
    int it_blocks = (N * C * num_tiles * 16 + 255) / 256;
    winograd_input_transform_kernel<<<it_blocks, 256, 0, stream>>>(
        input, input_tiles, N, C, H, W, num_tiles_h, num_tiles_w, out_H, out_W);
    
    int em_blocks = (N * out_C * num_tiles * 16 + 255) / 256;
    winograd_elementwise_kernel<<<em_blocks, 256, 0, stream>>>(
        input_tiles, weight_transformed, intermediate, N, C, out_C, num_tiles);
    
    int out_blocks = (N * out_C * out_H * out_W + 255) / 256;
    winograd_output_transform_kernel<<<out_blocks, 256, 0, stream>>>(
        intermediate, output, N, out_C, out_H, out_W, num_tiles_h, num_tiles_w);
    
    printf("[Winograd] Forward complete\n");
    CUDA_CHECK(cudaGetLastError());
}
