// tests/test_conv2d_cublas.cpp - Test cuBLAS-based Conv2d
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include "cuda_ops.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CPU reference implementation for conv2d forward
void cpu_conv2d_forward(const float* input, const float* weight, const float* bias,
                         float* output, int N, int C, int H, int W, int out_C,
                         int kernel_h, int kernel_w, int stride_h, int stride_w,
                         int pad_h, int pad_w) {
    int out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_C; ++oc) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    float sum = bias ? bias[oc] : 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride_h + kh - pad_h;
                                int iw = ow * stride_w + kw - pad_w;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int input_idx = n * C * H * W + c * H * W + ih * W + iw;
                                    int weight_idx = oc * C * kernel_h * kernel_w +
                                                      c * kernel_h * kernel_w + kh * kernel_w + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = n * out_C * out_H * out_W + oc * out_H * out_W + oh * out_W + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

TEST(Conv2dCublasTest, ForwardCorrectness) {
    // Test parameters
    int N = 2, C = 3, H = 8, W = 8;
    int out_C = 4, kernel_h = 3, kernel_w = 3;
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 1;

    int out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;  // 8
    int out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;  // 8

    // Allocate host memory
    float* h_input = new float[N * C * H * W];
    float* h_weight = new float[out_C * C * kernel_h * kernel_w];
    float* h_bias = new float[out_C];
    float* h_output_cublas = new float[N * out_C * out_H * out_W];
    float* h_output_cpu = new float[N * out_C * out_H * out_W];

    // Initialize input with random values
    for (int i = 0; i < N * C * H * W; ++i) {
        h_input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }
    for (int i = 0; i < out_C * C * kernel_h * kernel_w; ++i) {
        h_weight[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }
    for (int i = 0; i < out_C; ++i) {
        h_bias[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }

    // Allocate device memory
    float* d_input, *d_weight, *d_bias, *d_output;
    float* d_col_buffer, *d_gemm_buffer;
    CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, out_C * C * kernel_h * kernel_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * out_C * out_H * out_W * sizeof(float)));

    // Buffer sizes
    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;
    CUDA_CHECK(cudaMalloc(&d_col_buffer, col_rows * col_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm_buffer, out_C * col_cols * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, out_C * C * kernel_h * kernel_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, out_C * sizeof(float), cudaMemcpyHostToDevice));

    // Run cuBLAS conv2d
    cuda_conv2d_im2col_cublas(d_input, d_weight, d_bias, d_output,
                               d_col_buffer, d_gemm_buffer,
                               N, C, H, W, out_C, kernel_h, kernel_w,
                               stride_h, stride_w, pad_h, pad_w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output_cublas, d_output, N * out_C * out_H * out_W * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // CPU reference
    cpu_conv2d_forward(h_input, h_weight, h_bias, h_output_cpu,
                       N, C, H, W, out_C, kernel_h, kernel_w,
                       stride_h, stride_w, pad_h, pad_w);

    // Compare results
    float max_error = 0.0f;
    for (int i = 0; i < N * out_C * out_H * out_W; ++i) {
        float error = fabs(h_output_cublas[i] - h_output_cpu[i]);
        max_error = std::max(max_error, error);
    }

    printf("Conv2d cuBLAS forward max error: %.6f\n", max_error);
    EXPECT_LT(max_error, 1e-4f);

    // Cleanup
    delete[] h_input;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_output_cublas;
    delete[] h_output_cpu;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_col_buffer));
    CUDA_CHECK(cudaFree(d_gemm_buffer));
}

TEST(Conv2dCublasTest, BackwardCorrectness) {
    // Test parameters
    int N = 1, C = 2, H = 4, W = 4;
    int out_C = 2, kernel_h = 3, kernel_w = 3;
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 1;

    int out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;  // 4
    int out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;  // 4

    // Allocate host memory
    float* h_grad_output = new float[N * out_C * out_H * out_W];
    float* h_input = new float[N * C * H * W];
    float* h_weight = new float[out_C * C * kernel_h * kernel_w];
    float* h_grad_input_cublas = new float[N * C * H * W];
    float* h_grad_weight_cublas = new float[out_C * C * kernel_h * kernel_w];
    float* h_grad_bias_cublas = new float[out_C];

    // Initialize with simple values for verification
    for (int i = 0; i < N * out_C * out_H * out_W; ++i) {
        h_grad_output[i] = 1.0f;
    }
    for (int i = 0; i < N * C * H * W; ++i) {
        h_input[i] = (float)i * 0.1f;
    }
    for (int i = 0; i < out_C * C * kernel_h * kernel_w; ++i) {
        h_weight[i] = 0.5f;
    }

    // Allocate device memory
    float* d_grad_output, *d_input, *d_weight;
    float* d_grad_input, *d_grad_weight, *d_grad_bias;
    float* d_col_buffer, *d_grad_col_buffer, *d_grad_gemm_buffer;

    CUDA_CHECK(cudaMalloc(&d_grad_output, N * out_C * out_H * out_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, out_C * C * kernel_h * kernel_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weight, out_C * C * kernel_h * kernel_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias, out_C * sizeof(float)));

    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;
    CUDA_CHECK(cudaMalloc(&d_col_buffer, col_rows * col_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_col_buffer, col_rows * col_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gemm_buffer, out_C * col_cols * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, N * out_C * out_H * out_W * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, out_C * C * kernel_h * kernel_w * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Run backward
    cuda_conv2d_im2col_cublas_backward(d_grad_output, d_input, d_weight,
                                        d_grad_input, d_grad_weight, d_grad_bias,
                                        d_col_buffer, d_grad_col_buffer, d_grad_gemm_buffer,
                                        N, C, H, W, out_C, kernel_h, kernel_w,
                                        stride_h, stride_w, pad_h, pad_w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_grad_input_cublas, d_grad_input, N * C * H * W * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_weight_cublas, d_grad_weight,
                          out_C * C * kernel_h * kernel_w * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias_cublas, d_grad_bias, out_C * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify grad_bias (should be sum of grad_output for each output channel)
    printf("Grad bias: ");
    for (int oc = 0; oc < out_C; ++oc) {
        float expected_bias = N * out_H * out_W * 1.0f;  // all grad_output = 1.0
        printf("%.2f ", h_grad_bias_cublas[oc]);
        EXPECT_NEAR(h_grad_bias_cublas[oc], expected_bias, 1e-3f);
    }
    printf("\n");

    // grad_weight should be non-zero
    float weight_sum = 0.0f;
    for (int i = 0; i < out_C * C * kernel_h * kernel_w; ++i) {
        weight_sum += fabs(h_grad_weight_cublas[i]);
    }
    printf("Grad weight sum: %.4f\n", weight_sum);
    EXPECT_GT(weight_sum, 0.0f);

    // Cleanup
    delete[] h_grad_output;
    delete[] h_input;
    delete[] h_weight;
    delete[] h_grad_input_cublas;
    delete[] h_grad_weight_cublas;
    delete[] h_grad_bias_cublas;
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_weight));
    CUDA_CHECK(cudaFree(d_grad_bias));
    CUDA_CHECK(cudaFree(d_col_buffer));
    CUDA_CHECK(cudaFree(d_grad_col_buffer));
    CUDA_CHECK(cudaFree(d_grad_gemm_buffer));
}

TEST(Conv2dCublasTest, CompareWithIm2col) {
    // Compare cuBLAS version with existing im2col version
    int N = 2, C = 4, H = 16, W = 16;
    int out_C = 8, kernel_h = 3, kernel_w = 3;
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 1;

    int out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    // Allocate host memory
    float* h_input = new float[N * C * H * W];
    float* h_weight = new float[out_C * C * kernel_h * kernel_w];
    float* h_bias = new float[out_C];
    float* h_output_cublas = new float[N * out_C * out_H * out_W];
    float* h_output_im2col = new float[N * out_C * out_H * out_W];

    // Initialize
    for (int i = 0; i < N * C * H * W; ++i) h_input[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < out_C * C * kernel_h * kernel_w; ++i) h_weight[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < out_C; ++i) h_bias[i] = (float)(rand() % 100) / 100.0f;

    // Allocate device memory
    float* d_input, *d_weight, *d_bias;
    float* d_output_cublas, *d_output_im2col;
    float* d_col_buffer, *d_gemm_buffer;

    CUDA_CHECK(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, out_C * C * kernel_h * kernel_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_cublas, N * out_C * out_H * out_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_im2col, N * out_C * out_H * out_W * sizeof(float)));

    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;
    CUDA_CHECK(cudaMalloc(&d_col_buffer, col_rows * col_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm_buffer, out_C * col_cols * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, out_C * C * kernel_h * kernel_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, out_C * sizeof(float), cudaMemcpyHostToDevice));

    // Run cuBLAS version
    cuda_conv2d_im2col_cublas(d_input, d_weight, d_bias, d_output_cublas,
                               d_col_buffer, d_gemm_buffer,
                               N, C, H, W, out_C, kernel_h, kernel_w,
                               stride_h, stride_w, pad_h, pad_w);
    CUDA_CHECK(cudaGetLastError());

    // Run standard im2col version
    Conv2dDesc desc;
    desc.N = N; desc.C = C; desc.H = H; desc.W = W;
    desc.out_C = out_C; desc.kernel_h = kernel_h; desc.kernel_w = kernel_w;
    desc.stride_h = stride_h; desc.stride_w = stride_w;
    desc.pad_h = pad_h; desc.pad_w = pad_w;
    desc.groups = 1;
    desc.out_H = out_H; desc.out_W = out_W;

    cuda_conv2d_im2col(d_input, d_weight, d_bias, d_output_im2col,
                       d_col_buffer, d_gemm_buffer, desc, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_cublas, d_output_cublas, N * out_C * out_H * out_W * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_im2col, d_output_im2col, N * out_C * out_H * out_W * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Compare
    float max_error = 0.0f;
    for (int i = 0; i < N * out_C * out_H * out_W; ++i) {
        float error = fabs(h_output_cublas[i] - h_output_im2col[i]);
        max_error = std::max(max_error, error);
    }

    printf("cuBLAS vs im2col max error: %.6f\n", max_error);
    EXPECT_LT(max_error, 1e-5f);

    // Cleanup
    delete[] h_input;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_output_cublas;
    delete[] h_output_im2col;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output_cublas));
    CUDA_CHECK(cudaFree(d_output_im2col));
    CUDA_CHECK(cudaFree(d_col_buffer));
    CUDA_CHECK(cudaFree(d_gemm_buffer));
}