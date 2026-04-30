#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>
#include <cstdio>

class Conv2dWinogradF6Test : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(Conv2dWinogradF6Test, BasicCorrectness) {
    // Test basic correctness: N=1, C=1, H=8, W=8, out_C=1
    // Input: all 1s, Weight: all 1s
    // Expected output: 3x3 conv with stride=1, pad=1 means output is (H-2)x(W-2) = 6x6
    // For all-ones input and all-ones 3x3 kernel, each output = 9

    int N=1, C=1, H=8, W=8, out_C=1, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 6
    int out_W = W - 2;  // 6

    printf("\n=== Winograd F6 Basic Test ===\n");
    printf("Input: N=%d, C=%d, H=%d, W=%d\n", N, C, H, W);
    printf("Output: out_C=%d, out_H=%d, out_W=%d\n", out_C, out_H, out_W);

    std::vector<float> input(N * C * H * W, 1.0f);
    std::vector<float> weight(out_C * C * K * K, 1.0f);

    CudaBuffer d_input(N * C * H * W);
    CudaBuffer d_weight(out_C * C * K * K);
    CudaBuffer d_output(N * out_C * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate temp buffer size
    int num_tiles_h = (out_H + 5) / 6;  // 1
    int num_tiles_w = (out_W + 5) / 6;  // 1
    int num_tiles = num_tiles_h * num_tiles_w;

    size_t temp_size = out_C * C * 64 + N * C * num_tiles * 64 + N * out_C * num_tiles * 64;
    CudaBuffer d_temp(temp_size);

    printf("temp_size: %zu (weights=%d, input_tiles=%d, intermediate=%d)\n",
           temp_size, out_C*C*64, N*C*num_tiles*64, N*out_C*num_tiles*64);

    // Run Winograd F6
    cuda_conv2d_winograd_f6_forward(d_input.data, d_weight.data, nullptr,
                                     d_output.data, d_temp.data,
                                     N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get output
    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);

    printf("\nWinograd F6 output (6x6), expected all 9.0:\n");
    for (int i = 0; i < out_H; ++i) {
        for (int j = 0; j < out_W; ++j) {
            printf("%6.2f ", output[i * out_W + j]);
        }
        printf("\n");
    }

    // Compare with im2col baseline
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

    printf("\nim2col output:\n");
    for (int i = 0; i < out_H; ++i) {
        for (int j = 0; j < out_W; ++j) {
            printf("%6.2f ", output_im2col[i * out_W + j]);
        }
        printf("\n");
    }

    // Verify they match
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], output_im2col[i], 0.01f) << "Mismatch at index " << i;
    }
}

TEST_F(Conv2dWinogradF6Test, MultiChannel) {
    // Test with multiple input and output channels
    int N=1, C=2, H=8, W=8, out_C=2, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 6
    int out_W = W - 2;  // 6

    printf("\n=== Winograd F6 Multi-Channel Test ===\n");

    std::vector<float> input(N * C * H * W, 1.0f);
    std::vector<float> weight(out_C * C * K * K, 1.0f);

    CudaBuffer d_input(N * C * H * W);
    CudaBuffer d_weight(out_C * C * K * K);
    CudaBuffer d_output(N * out_C * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_tiles_h = (out_H + 5) / 6;
    int num_tiles_w = (out_W + 5) / 6;
    int num_tiles = num_tiles_h * num_tiles_w;

    size_t temp_size = out_C * C * 64 + N * C * num_tiles * 64 + N * out_C * num_tiles * 64;
    CudaBuffer d_temp(temp_size);

    cuda_conv2d_winograd_f6_forward(d_input.data, d_weight.data, nullptr,
                                     d_output.data, d_temp.data,
                                     N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);

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

    // For all-ones input with all-ones weight and 2 input channels:
    // Each output channel gets sum over all input channels of (3x3 conv on 1s) = 9 per channel
    // So output = 2 channels * 9 = 18 per output pixel

    printf("Winograd output (first channel, first row): ");
    for (int j = 0; j < out_W; ++j) {
        printf("%.2f ", output[j]);
    }
    printf("\n");

    printf("im2col output (first channel, first row): ");
    for (int j = 0; j < out_W; ++j) {
        printf("%.2f ", output_im2col[j]);
    }
    printf("\n");

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], output_im2col[i], 0.1f) << "Mismatch at index " << i;
    }
}

TEST_F(Conv2dWinogradF6Test, LargerInput) {
    // Test with larger input that spans multiple tiles
    int N=1, C=1, H=14, W=14, out_C=1, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 12
    int out_W = W - 2;  // 12

    printf("\n=== Winograd F6 Larger Input Test ===\n");
    printf("Input: %dx%d, Output: %dx%d\n", H, W, out_H, out_W);
    printf("Tiles needed: %dx%d\n", (out_H+5)/6, (out_W+5)/6);

    std::vector<float> input(N * C * H * W, 1.0f);
    std::vector<float> weight(out_C * C * K * K, 1.0f);

    CudaBuffer d_input(N * C * H * W);
    CudaBuffer d_weight(out_C * C * K * K);
    CudaBuffer d_output(N * out_C * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_tiles_h = (out_H + 5) / 6;  // (12+5)/6 = 2
    int num_tiles_w = (out_W + 5) / 6;  // (12+5)/6 = 2
    int num_tiles = num_tiles_h * num_tiles_w;

    size_t temp_size = out_C * C * 64 + N * C * num_tiles * 64 + N * out_C * num_tiles * 64;
    CudaBuffer d_temp(temp_size);

    cuda_conv2d_winograd_f6_forward(d_input.data, d_weight.data, nullptr,
                                     d_output.data, d_temp.data,
                                     N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);

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

    // With all-ones input and weight, all outputs should be 9
    printf("Winograd output (6x6, center):\n");
    for (int i = 3; i < 9; ++i) {
        for (int j = 3; j < 9; ++j) {
            printf("%6.2f ", output[i * out_W + j]);
        }
        printf("\n");
    }

    int max_err_count = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float err = std::abs(output[i] - output_im2col[i]);
        if (err > max_err) {
            max_err = err;
        }
        if (err > 0.01f && max_err_count < 5) {
            printf("Mismatch at index %zu: winograd=%.2f, im2col=%.2f\n",
                   i, output[i], output_im2col[i]);
            max_err_count++;
        }
    }
    printf("Max error: %.6f\n", max_err);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], output_im2col[i], 0.1f) << "Mismatch at index " << i;
    }
}

TEST_F(Conv2dWinogradF6Test, NonUniformInput) {
    // Test with non-uniform input to verify transform correctness
    int N=1, C=1, H=8, W=8, out_C=1, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 6
    int out_W = W - 2;  // 6

    // Input: incrementing values
    std::vector<float> input(N * C * H * W);
    for (int i = 0; i < N * C * H * W; ++i) {
        input[i] = static_cast<float>(i);
    }

    // Weight: [1, 2, 3; 4, 5, 6; 7, 8, 9]
    std::vector<float> weight = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    CudaBuffer d_input(N * C * H * W);
    CudaBuffer d_weight(out_C * C * K * K);
    CudaBuffer d_output(N * out_C * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_tiles_h = (out_H + 5) / 6;
    int num_tiles_w = (out_W + 5) / 6;
    int num_tiles = num_tiles_h * num_tiles_w;

    size_t temp_size = out_C * C * 64 + N * C * num_tiles * 64 + N * out_C * num_tiles * 64;
    CudaBuffer d_temp(temp_size);

    cuda_conv2d_winograd_f6_forward(d_input.data, d_weight.data, nullptr,
                                     d_output.data, d_temp.data,
                                     N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);

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

    printf("\n=== Non-uniform input test ===\n");
    printf("Winograd output:\n");
    for (int i = 0; i < out_H; ++i) {
        for (int j = 0; j < out_W; ++j) {
            printf("%8.2f ", output[i * out_W + j]);
        }
        printf("\n");
    }

    printf("\nim2col output:\n");
    for (int i = 0; i < out_H; ++i) {
        for (int j = 0; j < out_W; ++j) {
            printf("%8.2f ", output_im2col[i * out_W + j]);
        }
        printf("\n");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float err = std::abs(output[i] - output_im2col[i]);
        if (err > max_err) max_err = err;
        // Winograd F(6×6) uses large transform values (up to 3125), causing amplified FP errors
        // Relative error < 0.1% is acceptable
        EXPECT_NEAR(output[i], output_im2col[i], 2.0f) << "Mismatch at index " << i;
    }
    printf("Max error: %.6f\n", max_err);
}