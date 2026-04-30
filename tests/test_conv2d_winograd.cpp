#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>
#include <cstdio>

class WinogradDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(WinogradDebugTest, DebugTileMapping) {
    // Minimal case: N=1, C=1, H=4, W=4, out_C=1, K=3, stride=1, pad=1
    // Input: all 1s
    // Weight: all 1s
    // Expected output for 3x3 conv with all-ones: 9 for each output pixel
    
    int N=1, C=1, H=4, W=4, out_C=1, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 2
    int out_W = W - 2;  // 2
    
    printf("\n=== Debug Tile Mapping ===\n");
    printf("Input: N=%d, C=%d, H=%d, W=%d\n", N, C, H, W);
    printf("Output: out_C=%d, out_H=%d, out_W=%d\n", out_C, out_H, out_W);
    printf("Tiles: num_tiles_h=%d, num_tiles_w=%d\n", (out_H+1)/2, (out_W+1)/2);
    
    // Input all 1s
    std::vector<float> input(N * C * H * W, 1.0f);
    // Weight all 1s
    std::vector<float> weight(out_C * C * K * K, 1.0f);
    
    CudaBuffer d_input(N * C * H * W);
    CudaBuffer d_weight(out_C * C * K * K);
    CudaBuffer d_output(N * out_C * out_H * out_W);
    
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate temp buffer
    int num_tiles_h = (out_H + 1) / 2;  // 1
    int num_tiles_w = (out_W + 1) / 2;  // 1
    int num_tiles = num_tiles_h * num_tiles_w;
    
    size_t temp_size = out_C * C * 16 + N * C * num_tiles * 16 + N * out_C * num_tiles * 16;
    CudaBuffer d_temp(temp_size);
    
    printf("temp_size: %zu (weights=%d, input_tiles=%d, intermediate=%d)\n",
           temp_size, out_C*C*16, N*C*num_tiles*16, N*out_C*num_tiles*16);
    
    // Run Winograd
    cuda_conv2d_winograd_forward(d_input.data, d_weight.data, nullptr,
                                  d_output.data, d_temp.data,
                                  N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check output
    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);
    
    printf("\nWinograd output: ");
    for (size_t i = 0; i < output.size(); ++i) {
        printf("%.2f ", output[i]);
    }
    printf("\n");
    
    printf("\nExpected (im2col): 4.0 6.0 / 6.0 9.0\n");
    
    // Compare with im2col
    CudaBuffer d_output_im2col(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * K * K * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);
    
    Conv2dDesc desc;
    desc.N = N; desc.C = C; desc.H = H; desc.W = W;
    desc.out_C = out_C; desc.out_H = out_H; desc.out_W = out_W;
    desc.kernel_h = K; desc.kernel_w = K;
    desc.stride_h = stride; desc.stride_w = stride;
    desc.pad_h = pad; desc.pad_w = pad;
    desc.groups = 1;
    
    cuda_conv2d_im2col(d_input.data, d_weight.data, nullptr,
                       d_output_im2col.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> output_im2col(N * out_C * out_H * out_W);
    device_to_host(d_output_im2col.data, output_im2col.data(), N * out_C * out_H * out_W);
    
    printf("im2col output:  ");
    for (size_t i = 0; i < output_im2col.size(); ++i) {
        printf("%.2f ", output_im2col[i]);
    }
    printf("\n");
    
    // Verify Winograd matches im2col
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], output_im2col[i], 0.01f);
    }
}

TEST_F(WinogradDebugTest, SimpleConvReference) {
    // Simple test with a known convolution result
    int N=1, C=1, H=5, W=5, out_C=1, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 3
    int out_W = W - 2;  // 3
    
    // Input: [0,1,2,3,4]
    //        [5,6,7,8,9]
    //        [10,11,12,13,14]
    //        [15,16,17,18,19]
    //        [20,21,22,23,24]
    std::vector<float> input = {
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
    };
    
    // Weight: all ones (3x3)
    std::vector<float> weight = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    // Output position (0,0) = sum of input[0:3, 0:3] = 0+1+2+5+6+7+10+11+12 = 54
    // Output position (1,1) = sum of input[1:4, 1:4] = 6+7+8+11+12+13+16+17+18 = 108
    
    CudaBuffer d_input(N * C * H * W);
    CudaBuffer d_weight(out_C * C * K * K);
    
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // im2col baseline
    CudaBuffer d_output_im2col(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * K * K * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);
    
    Conv2dDesc desc;
    desc.N = N; desc.C = C; desc.H = H; desc.W = W;
    desc.out_C = out_C; desc.out_H = out_H; desc.out_W = out_W;
    desc.kernel_h = K; desc.kernel_w = K;
    desc.stride_h = stride; desc.stride_w = stride;
    desc.pad_h = pad; desc.pad_w = pad;
    desc.groups = 1;
    
    cuda_conv2d_im2col(d_input.data, d_weight.data, nullptr,
                       d_output_im2col.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> output_im2col(N * out_C * out_H * out_W);
    device_to_host(d_output_im2col.data, output_im2col.data(), N * out_C * out_H * out_W);
    
    printf("\nSimple conv reference:\n");
    printf("Input (5x5):\n");
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            printf("%3.0f ", input[i * W + j]);
        }
        printf("\n");
    }
    
    printf("\nExpected output (3x3) - sum of 3x3 windows:\n");
    for (int oh = 0; oh < out_H; ++oh) {
        for (int ow = 0; ow < out_W; ++ow) {
            float sum = 0;
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    sum += input[(oh + kh) * W + (ow + kw)];
                }
            }
            printf("%3.0f ", sum);
        }
        printf("\n");
    }
    
    printf("\nim2col output:\n");
    for (int oh = 0; oh < out_H; ++oh) {
        for (int ow = 0; ow < out_W; ++ow) {
            printf("%6.2f ", output_im2col[oh * out_W + ow]);
        }
        printf("\n");
    }
}
