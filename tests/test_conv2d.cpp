#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

class Conv2dTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    std::vector<float> generate_random(size_t size) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = -1.0f + static_cast<float>(rand()) / RAND_MAX * 2.0f;
        }
        return v;
    }
};

TEST_F(Conv2dTest, Basic) {
    Conv2dDesc desc;
    desc.N = 1; desc.C = 1; desc.H = 5; desc.W = 5;
    desc.out_C = 1; desc.out_H = 5; desc.out_W = 5;
    desc.kernel_h = 3; desc.kernel_w = 3;
    desc.stride_h = 1; desc.stride_w = 1;
    desc.pad_h = 1; desc.pad_w = 1;
    desc.groups = 1;

    std::vector<float> input = {{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    }};

    std::vector<float> weight = {{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    }};

    std::vector<float> bias = {0.0f};

    std::vector<float> output_ref(desc.N * desc.out_C * desc.out_H * desc.out_W, 0.0f);

    for (int n = 0; n < desc.N; ++n) {
        for (int oc = 0; oc < desc.out_C; ++oc) {
            for (int oh = 0; oh < desc.out_H; ++oh) {
                for (int ow = 0; ow < desc.out_W; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < desc.kernel_h; ++kh) {
                        for (int kw = 0; kw < desc.kernel_w; ++kw) {
                            int ih = oh + kh - desc.pad_h;
                            int iw = ow + kw - desc.pad_w;
                            if (ih >= 0 && ih < desc.H && iw >= 0 && iw < desc.W) {
                                int in_idx = n * desc.C * desc.H * desc.W + 0 * desc.H * desc.W + ih * desc.W + iw;
                                int w_idx = oc * desc.C * desc.kernel_h * desc.kernel_w + 0 * desc.kernel_h * desc.kernel_w + kh * desc.kernel_w + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    int out_idx = n * desc.out_C * desc.out_H * desc.out_W + oc * desc.out_H * desc.out_W + oh * desc.out_W + ow;
                    output_ref[out_idx] = sum + bias[oc];
                }
            }
        }
    }

    CudaBuffer d_input(desc.N * desc.C * desc.H * desc.W),
               d_weight(desc.out_C * desc.C * desc.kernel_h * desc.kernel_w),
               d_bias(desc.out_C),
               d_output(desc.N * desc.out_C * desc.out_H * desc.out_W);

    host_to_device_async(d_input.data, input.data(), input.size());
    host_to_device_async(d_weight.data, weight.data(), weight.size());
    host_to_device_async(d_bias.data, bias.data(), bias.size());

    cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);

    std::vector<float> output(desc.N * desc.out_C * desc.out_H * desc.out_W);
    device_to_host(d_output.data, output.data(), output.size());

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], output_ref[i], 1e-4f);
    }
}

TEST_F(Conv2dTest, NoPadding) {
    Conv2dDesc desc;
    desc.N = 1; desc.C = 1; desc.H = 4; desc.W = 4;
    desc.out_C = 1; desc.out_H = 2; desc.out_W = 2;
    desc.kernel_h = 2; desc.kernel_w = 2;
    desc.stride_h = 2; desc.stride_w = 2;
    desc.pad_h = 0; desc.pad_w = 0;
    desc.groups = 1;

    std::vector<float> input = {{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }};

    std::vector<float> weight = {{
        1, 0,
        0, 1
    }};

    std::vector<float> bias = {0.0f};

    std::vector<float> output_ref(desc.N * desc.out_C * desc.out_H * desc.out_W, 0.0f);

    for (int n = 0; n < desc.N; ++n) {
        for (int oc = 0; oc < desc.out_C; ++oc) {
            for (int oh = 0; oh < desc.out_H; ++oh) {
                for (int ow = 0; ow < desc.out_W; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < desc.kernel_h; ++kh) {
                        for (int kw = 0; kw < desc.kernel_w; ++kw) {
                            int ih = oh * desc.stride_h + kh;
                            int iw = ow * desc.stride_w + kw;
                            int in_idx = n * desc.C * desc.H * desc.W + 0 * desc.H * desc.W + ih * desc.W + iw;
                            int w_idx = oc * desc.C * desc.kernel_h * desc.kernel_w + kh * desc.kernel_w + kw;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                    int out_idx = n * desc.out_C * desc.out_H * desc.out_W + oc * desc.out_H * desc.out_W + oh * desc.out_W + ow;
                    output_ref[out_idx] = sum + bias[oc];
                }
            }
        }
    }

    CudaBuffer d_input(desc.N * desc.C * desc.H * desc.W),
               d_weight(desc.out_C * desc.C * desc.kernel_h * desc.kernel_w),
               d_bias(desc.out_C),
               d_output(desc.N * desc.out_C * desc.out_H * desc.out_W);

    host_to_device_async(d_input.data, input.data(), input.size());
    host_to_device_async(d_weight.data, weight.data(), weight.size());
    host_to_device_async(d_bias.data, bias.data(), bias.size());

    cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);

    std::vector<float> output(desc.N * desc.out_C * desc.out_H * desc.out_W);
    device_to_host(d_output.data, output.data(), output.size());

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], output_ref[i], 1e-4f);
    }
}

TEST_F(Conv2dTest, BackwardPass) {
    int N = 1, C = 3, H = 8, W = 8;
    int out_C = 4, kernel_h = 3, kernel_w = 3;
    int stride = 1, pad = 1;
    int out_H = (H + 2 * pad - kernel_h) / stride + 1;
    int out_W = (W + 2 * pad - kernel_w) / stride + 1;

    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * kernel_h * kernel_w);
    auto bias = generate_random(out_C);
    auto grad_out = generate_random(N * out_C * out_H * out_W);

    // Reference backward (simplified version - only verifying grad_bias)
    std::vector<float> grad_bias_ref(out_C, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_C; ++oc) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    grad_bias_ref[oc] += grad_out[n * out_C * out_H * out_W +
                                                  oc * out_H * out_W +
                                                  oh * out_W + ow];
                }
            }
        }
    }

    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * kernel_h * kernel_w);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    CudaBuffer d_grad_out(N * out_C * out_H * out_W);
    CudaBuffer d_grad_input(N * C * H * W), d_grad_weight(out_C * C * kernel_h * kernel_w);
    CudaBuffer d_grad_bias(out_C);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * kernel_h * kernel_w);
    host_to_device_async(d_bias.data, bias.data(), out_C);
    host_to_device_async(d_grad_out.data, grad_out.data(), N * out_C * out_H * out_W);

    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W,
                    kernel_h, kernel_w, stride, stride, pad, pad, 1};

    cuda_conv2d_backward(d_grad_out.data, d_input.data, d_weight.data,
                         d_grad_input.data, d_grad_weight.data, d_grad_bias.data,
                         desc);

    std::vector<float> grad_bias(out_C);
    device_to_host(d_grad_bias.data, grad_bias.data(), out_C);

    for (int oc = 0; oc < out_C; ++oc) {
        EXPECT_NEAR(grad_bias[oc], grad_bias_ref[oc], 1e-3f);
    }
}

TEST_F(Conv2dTest, Im2colCorrectness) {
    int N = 2, C = 4, H = 8, W = 8;
    int out_C = 8, kernel = 3, stride = 1, pad = 1;
    int out_H = (H + 2 * pad - kernel) / stride + 1;
    int out_W = (W + 2 * pad - kernel) / stride + 1;

    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * kernel * kernel);
    auto bias = generate_random(out_C);

    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W, kernel, kernel, stride, stride, pad, pad, 1};

    // Allocate buffers
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * kernel * kernel);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    CudaBuffer d_output_im2col(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * kernel * kernel * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * kernel * kernel);
    host_to_device_async(d_bias.data, bias.data(), out_C);

    // Naive conv2d
    cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // im2col + GEMM conv2d
    cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                       d_output_im2col.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output_naive(N * out_C * out_H * out_W);
    std::vector<float> output_im2col(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output_naive.data(), N * out_C * out_H * out_W);
    device_to_host(d_output_im2col.data, output_im2col.data(), N * out_C * out_H * out_W);

    // Compare results
    for (size_t i = 0; i < output_naive.size(); ++i) {
        EXPECT_NEAR(output_naive[i], output_im2col[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST_F(Conv2dTest, Im2colCorrectnessLarge) {
    int N = 4, C = 16, H = 16, W = 16;
    int out_C = 32, kernel = 3, stride = 1, pad = 1;
    int out_H = (H + 2 * pad - kernel) / stride + 1;
    int out_W = (W + 2 * pad - kernel) / stride + 1;

    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * kernel * kernel);
    auto bias = generate_random(out_C);

    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W, kernel, kernel, stride, stride, pad, pad, 1};

    // Allocate buffers
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * kernel * kernel);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    CudaBuffer d_output_im2col(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * kernel * kernel * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * kernel * kernel);
    host_to_device_async(d_bias.data, bias.data(), out_C);

    // Naive conv2d
    cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // im2col + GEMM conv2d
    cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                       d_output_im2col.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output_naive(N * out_C * out_H * out_W);
    std::vector<float> output_im2col(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output_naive.data(), N * out_C * out_H * out_W);
    device_to_host(d_output_im2col.data, output_im2col.data(), N * out_C * out_H * out_W);

    // Compare results
    for (size_t i = 0; i < output_naive.size(); ++i) {
        EXPECT_NEAR(output_naive[i], output_im2col[i], 1e-3f)
            << "Mismatch at index " << i;
    }
}

TEST_F(Conv2dTest, PerformanceBenchmark) {
    int N = 32, C = 64, H = 32, W = 32;
    int out_C = 64, kernel = 3, stride = 1, pad = 1;
    int out_H = (H + 2 * pad - kernel) / stride + 1;
    int out_W = (W + 2 * pad - kernel) / stride + 1;

    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * kernel * kernel);
    auto bias = generate_random(out_C);

    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W, kernel, kernel, stride, stride, pad, pad, 1};

    // Allocate buffers
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * kernel * kernel);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * kernel * kernel * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * kernel * kernel);
    host_to_device_async(d_bias.data, bias.data(), out_C);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark naive
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(end1 - start1).count() / 100;

    // Warmup im2col
    for (int i = 0; i < 10; ++i) {
        cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                          d_output.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark im2col
    start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                          d_output.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end1 = std::chrono::high_resolution_clock::now();
    double im2col_ms = std::chrono::duration<double, std::milli>(end1 - start1).count() / 100;

    std::cout << "\n========== Conv2d Performance (N=32, C=64, H=W=32, K=3) ==========\n";
    std::cout << "  Naive:   " << naive_ms << " ms\n";
    std::cout << "  im2col:  " << im2col_ms << " ms\n";
    std::cout << "  Speedup: " << naive_ms / im2col_ms << "x\n";
    std::cout << "==================================================================\n";

    // Calculate GFLOPS
    // Each output element requires C * kernel_h * kernel_w multiplications and additions
    // Total FLOPs = 2 * N * out_C * out_H * out_W * C * kernel_h * kernel_w
    long long flops = 2LL * N * out_C * out_H * out_W * C * kernel * kernel;
    double naive_gflops = flops / (naive_ms * 1e6);
    double im2col_gflops = flops / (im2col_ms * 1e6);
    std::cout << "  Naive GFLOPS:   " << naive_gflops << "\n";
    std::cout << "  im2col GFLOPS:  " << im2col_gflops << "\n";
}