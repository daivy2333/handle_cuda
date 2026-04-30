#include "cuda_ops.h"
#include "cuda_util.h"
#include <cstdio>

namespace {

// Winograd F(6×6, 3×3) Transform Matrices (wincnn.cookToomFilter standard)
// Generated with interpolation points [0,1,2,3,4,5,-1]
// G: 3x3 weight -> 8x8 transform space (G @ w @ G^T)
__device__ __constant__ float winograd_G_f6[8][3] = {
    {0.008333f,  0.000000f,  0.000000f},
    {0.020833f,  0.020833f,  0.020833f},
    {-0.027778f, -0.055556f, -0.111111f},
    {0.020833f,  0.062500f,  0.187500f},
    {-0.008333f, -0.033333f, -0.133333f},
    {0.001389f,  0.006944f,  0.034722f},
    {0.001389f, -0.001389f,  0.001389f},
    {0.000000f,  0.000000f,  1.000000f}
};

// B^T: 8x8 input transform left side: B^T @ U @ B
__device__ __constant__ float winograd_Bt_f6[8][8] = {
    {120.0f, -154.0f,  -49.0f,  140.0f,  -70.0f,   14.0f,   -1.0f,   0.0f},
    {  0.0f,  120.0f,  -34.0f,  -83.0f,   57.0f,  -13.0f,    1.0f,   0.0f},
    {  0.0f,   60.0f,  -47.0f,  -48.0f,   46.0f,  -12.0f,    1.0f,   0.0f},
    {  0.0f,   40.0f,  -38.0f,  -29.0f,   37.0f,  -11.0f,    1.0f,   0.0f},
    {  0.0f,   30.0f,  -31.0f,  -20.0f,   30.0f,  -10.0f,    1.0f,   0.0f},
    {  0.0f,   24.0f,  -26.0f,  -15.0f,   25.0f,   -9.0f,    1.0f,   0.0f},
    {  0.0f, -120.0f,  274.0f, -225.0f,   85.0f,  -15.0f,    1.0f,   0.0f},
    {  0.0f, -120.0f,  154.0f,   49.0f, -140.0f,   70.0f,  -14.0f,   1.0f}
};

// B = B^T^T (right side of input transform)
__device__ __constant__ float winograd_B_f6[8][8] = {
    {120.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f},
    {-154.0f,  120.0f,   60.0f,   40.0f,   30.0f,   24.0f, -120.0f, -120.0f},
    { -49.0f,  -34.0f,  -47.0f,  -38.0f,  -31.0f,  -26.0f,  274.0f,  154.0f},
    { 140.0f,  -83.0f,  -48.0f,  -29.0f,  -20.0f,  -15.0f, -225.0f,   49.0f},
    { -70.0f,   57.0f,   46.0f,   37.0f,   30.0f,   25.0f,   85.0f, -140.0f},
    {  14.0f,  -13.0f,  -12.0f,  -11.0f,  -10.0f,   -9.0f,  -15.0f,   70.0f},
    {  -1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,  -14.0f},
    {   0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    0.0f,    1.0f}
};

// A^T: 6x8 output transform: A^T @ M @ A
__device__ __constant__ float winograd_At_f6[6][8] = {
    {1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,  0.0f},
    {0.0f,    1.0f,    2.0f,    3.0f,    4.0f,    5.0f,   -1.0f,  0.0f},
    {0.0f,    1.0f,    4.0f,    9.0f,   16.0f,   25.0f,    1.0f,  0.0f},
    {0.0f,    1.0f,    8.0f,   27.0f,   64.0f,  125.0f,   -1.0f,  0.0f},
    {0.0f,    1.0f,   16.0f,   81.0f,  256.0f,  625.0f,    1.0f,  0.0f},
    {0.0f,    1.0f,   32.0f,  243.0f, 1024.0f, 3125.0f,   -1.0f,  1.0f}
};

// Kernel to transform 3x3 weight to 8x8: W = G @ w @ G^T
__global__ void winograd_f6_weight_transform_kernel(
    const float* weight, float* transformed, int out_C, int C) {

    int oc = blockIdx.x;
    int c = blockIdx.y;
    if (oc >= out_C || c >= C) return;

    float w_local[3][3];
    float W_local[8][8] = {{0}};

    // Load 3x3 weight
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            w_local[i][j] = weight[(oc * C + c) * 9 + i * 3 + j];
        }
    }

    // Compute W = G @ w @ G^T (8x8 result)
    // First: temp = G @ w (8x3)
    float temp[8][3] = {{0}};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                temp[i][j] += winograd_G_f6[i][k] * w_local[k][j];
            }
        }
    }

    // Second: W = temp @ G^T (8x8)
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                W_local[i][j] += temp[i][k] * winograd_G_f6[j][k];
            }
        }
    }

    // Store transformed weight (8x8)
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            transformed[(oc * C + c) * 64 + i * 8 + j] = W_local[i][j];
        }
    }
}

// Kernel to transform 8x8 input tile: V = B^T @ U @ B
__global__ void winograd_f6_input_transform_kernel(
    const float* input, float* transformed,
    int N, int C, int H, int W,
    int num_tiles_h, int num_tiles_w) {

    int n = blockIdx.z;
    int c = blockIdx.y;
    int tile_idx = blockIdx.x;

    if (n >= N || c >= C) return;

    int tile_h = tile_idx / num_tiles_w;
    int tile_w = tile_idx % num_tiles_w;

    if (tile_h >= num_tiles_h || tile_w >= num_tiles_w) return;

    // Extract 8x8 tile with padding
    float U[8][8] = {{0}};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int h = tile_h * 6 + i - 1;  // stride=6, pad=1
            int w = tile_w * 6 + j - 1;
            if (h >= 0 && h < H && w >= 0 && w < W) {
                U[i][j] = input[((n * C + c) * H + h) * W + w];
            }
        }
    }

    // Compute V = B^T @ U @ B
    float temp[8][8] = {{0}};
    float V[8][8] = {{0}};

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                temp[i][j] += winograd_Bt_f6[i][k] * U[k][j];
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                V[i][j] += temp[i][k] * winograd_B_f6[k][j];
            }
        }
    }

    // Store V (8x8)
    int tile_id = (n * C + c) * num_tiles_h * num_tiles_w + tile_idx;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            transformed[tile_id * 64 + i * 8 + j] = V[i][j];
        }
    }
}

// Element-wise multiply kernel: M = V ⊙ W (sum over input channels)
__global__ void winograd_f6_elementwise_kernel(
    const float* transformed_weights, const float* transformed_inputs,
    float* intermediate,
    int N, int out_C, int C, int num_tiles) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * num_tiles * 64;  // 64 = 8*8
    if (idx >= total) return;

    int rem = idx;
    int n = rem / (out_C * num_tiles * 64);
    rem %= (out_C * num_tiles * 64);
    int oc = rem / (num_tiles * 64);
    rem %= (num_tiles * 64);
    int tile_idx = rem / 64;
    int elem_idx = rem % 64;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = transformed_inputs[((n * C + c) * num_tiles + tile_idx) * 64 + elem_idx];
        float w = transformed_weights[(oc * C + c) * 64 + elem_idx];
        sum += v * w;
    }

    intermediate[idx] = sum;
}

// Output transform kernel: Y = A^T @ M @ A, extracting 6x6 output from 8x8 M
__global__ void winograd_f6_output_transform_kernel(
    const float* intermediate, float* output,
    int N, int out_C, int out_H, int out_W,
    int num_tiles_h, int num_tiles_w) {

    int n = blockIdx.z;
    int oc = blockIdx.y;
    int tile_idx = blockIdx.x;

    if (n >= N || oc >= out_C) return;

    int tile_h = tile_idx / num_tiles_w;
    int tile_w = tile_idx % num_tiles_w;

    if (tile_h >= num_tiles_h || tile_w >= num_tiles_w) return;

    // Load 8x8 M tile
    float M[8][8];
    int tile_id = n * out_C * num_tiles_h * num_tiles_w + oc * num_tiles_h * num_tiles_w + tile_idx;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            M[i][j] = intermediate[tile_id * 64 + i * 8 + j];
        }
    }

    // Compute Y = A^T @ M @ A (6x6 output)
    float temp[6][8] = {{0}};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                temp[i][j] += winograd_At_f6[i][k] * M[k][j];
            }
        }
    }

    float Y[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 8; ++k) {
                sum += temp[i][k] * winograd_At_f6[j][k];
            }
            Y[i][j] = sum;
        }
    }

    // Store 6x6 output to correct position in output tensor
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            int out_h = tile_h * 6 + i;
            int out_w = tile_w * 6 + j;
            if (out_h < out_H && out_w < out_W) {
                output[((n * out_C + oc) * out_H + out_h) * out_W + out_w] = Y[i][j];
            }
        }
    }
}

} // namespace

void cuda_conv2d_winograd_f6_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_buffer,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream) {

    printf("[Winograd F6] Forward: N=%d, C=%d, H=%d, W=%d, out_C=%d\n", N, C, H, W, out_C);

    if (stride_h != 1 || stride_w != 1) {
        printf("[Winograd F6] ERROR: Only supports stride=1\n");
        return;
    }

    if (pad_h != 1 || pad_w != 1) {
        printf("[Winograd F6] ERROR: Only supports pad=1\n");
        return;
    }

    // Winograd F(6×6) with 3×3 kernel: output = (H - 2) × (W - 2)
    int out_H = H - 2;
    int out_W = W - 2;

    // Calculate number of tiles (each tile produces 6x6 output)
    int num_tiles_h = (out_H + 5) / 6;
    int num_tiles_w = (out_W + 5) / 6;
    int num_tiles = num_tiles_h * num_tiles_w;

    printf("[Winograd F6] out_H=%d, out_W=%d, tiles=%dx%d (%d total)\n",
           out_H, out_W, num_tiles_h, num_tiles_w, num_tiles);

    // Layout in temp_buffer:
    // [0, out_C*C*64): transformed weights [out_C, C, 8, 8]
    // [out_C*C*64, out_C*C*64 + N*C*num_tiles*64): transformed inputs [N, C, num_tiles, 8, 8]
    // [out_C*C*64 + N*C*num_tiles*64, ...): intermediate [N, out_C, num_tiles, 8, 8]

    float* transformed_weights = temp_buffer;
    float* transformed_inputs = temp_buffer + out_C * C * 64;
    float* intermediate = temp_buffer + out_C * C * 64 + N * C * num_tiles * 64;

    // Step 1: Transform weights (G @ w @ G^T)
    dim3 wt_grid(out_C, C);
    winograd_f6_weight_transform_kernel<<<wt_grid, 1, 0, stream>>>(
        weight, transformed_weights, out_C, C);

    // Step 2: Transform inputs (B^T @ U @ B)
    dim3 it_grid(num_tiles, C, N);
    winograd_f6_input_transform_kernel<<<it_grid, 1, 0, stream>>>(
        input, transformed_inputs, N, C, H, W, num_tiles_h, num_tiles_w);

    // Step 3: Element-wise multiply and sum over input channels
    int elem_total = N * out_C * num_tiles * 64;
    int elem_block = 256;
    int elem_grid = (elem_total + elem_block - 1) / elem_block;
    winograd_f6_elementwise_kernel<<<elem_grid, elem_block, 0, stream>>>(
        transformed_weights, transformed_inputs, intermediate, N, out_C, C, num_tiles);

    // Step 4: Output transform (A^T @ M @ A)
    dim3 ot_grid(num_tiles, out_C, N);
    winograd_f6_output_transform_kernel<<<ot_grid, 1, 0, stream>>>(
        intermediate, output, N, out_C, out_H, out_W, num_tiles_h, num_tiles_w);

    printf("[Winograd F6] Forward complete\n");
    CUDA_CHECK(cudaGetLastError());
}