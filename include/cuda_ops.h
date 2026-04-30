#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

enum class ActivationType { ReLU, Sigmoid, Tanh, Softmax };

struct MatMulDesc {
    size_t M;
    size_t N;
    size_t K;
    bool transpose_a;
    bool transpose_b;
};

struct Conv2dDesc {
    int N, C, H, W;
    int out_C, out_H, out_W;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int groups;
};

struct Pool2dDesc {
    int N, C, H, W;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
};

// Pre-allocated buffers for conv2d backward (avoid malloc/free per batch)
struct Conv2dBackwardBuffers {
    float* reshaped_grad;   // [out_C, N*out_H*out_W]
    float* col_buffer;      // [C*K*K, N*out_H*out_W]
    float* col_grad;        // [C*K*K, N*out_H*out_W]
};

void cuda_matmul(const float* A, const float* B, float* C, const MatMulDesc& desc, cudaStream_t stream = 0);
void cuda_matmul_backward(const float* grad_C, const float* A, const float* B,
                          float* grad_A, float* grad_B,
                          const MatMulDesc& desc, cudaStream_t stream = 0);

void cuda_bias_add(const float* input, const float* bias, float* output, size_t rows, size_t cols, cudaStream_t stream = 0);
void cuda_bias_add_backward(const float* grad_out, float* grad_input,
                             float* grad_bias, size_t rows, size_t cols,
                             cudaStream_t stream = 0);

void cuda_relu(float* data, size_t size, cudaStream_t stream = 0);
void cuda_relu_backward(const float* grad_out, const float* forward_input, float* grad_in, size_t size, cudaStream_t stream = 0);

void cuda_sigmoid(float* data, size_t size, cudaStream_t stream = 0);
void cuda_sigmoid(const float* input, float* output, size_t size, cudaStream_t stream = 0);
void cuda_sigmoid_backward(const float* grad_out, const float* forward_output,
                           float* grad_in, size_t size, cudaStream_t stream = 0);

void cuda_tanh(float* data, size_t size, cudaStream_t stream = 0);
void cuda_tanh(const float* input, float* output, size_t size, cudaStream_t stream = 0);
void cuda_tanh_backward(const float* grad_out, const float* forward_output,
                        float* grad_in, size_t size, cudaStream_t stream = 0);

void cuda_softmax(const float* input, float* output, size_t batch_size, size_t num_classes, cudaStream_t stream = 0);
void cuda_softmax_backward(const float* grad_out, const float* forward_output, float* grad_in, size_t batch_size, size_t num_classes, cudaStream_t stream = 0);

void cuda_conv2d(const float* input, const float* weight, const float* bias, float* output,
                const Conv2dDesc& desc, cudaStream_t stream = 0);
void cuda_conv2d_im2col(const float* input, const float* weight, const float* bias,
                        float* output, float* col_buffer, float* gemm_buffer,
                        const Conv2dDesc& desc, cudaStream_t stream = 0);
void cuda_conv2d_backward(const float* grad_out, const float* input, const float* weight,
                          float* grad_input, float* grad_weight, float* grad_bias,
                          const Conv2dDesc& desc, cudaStream_t stream = 0);

// Optimized backward with pre-allocated buffers (no malloc/free overhead)
void cuda_conv2d_backward_with_buffers(const float* grad_out, const float* input, const float* weight,
                                        float* grad_input, float* grad_weight, float* grad_bias,
                                        const Conv2dDesc& desc,
                                        float* reshaped_grad, float* col_buffer, float* col_grad,
                                        cudaStream_t stream = 0);

// Calculate buffer sizes for conv2d backward
size_t cuda_conv2d_backward_buffer_sizes(const Conv2dDesc& desc,
                                          size_t* reshaped_size, size_t* col_size);

// Winograd F(4×4, 3×3) conv2d forward - 2-3x faster for 3x3 conv
void cuda_conv2d_winograd_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_buffer,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = 0);

// Fused Conv2d + ReLU kernel (reduces memory bandwidth)
void cuda_conv2d_relu_fused(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = 0);

// Fused Conv2d + ReLU + MaxPool2d kernel
void cuda_conv2d_relu_pool_fused(
    const float* input, const float* weight, const float* bias,
    float* output, int* max_indices,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = 0);

// FP16/Tensor Core matmul (requires compute capability 7.0+)
void cuda_matmul_fp16(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0);

void cuda_matmul_fp32_baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0);

void cuda_matmul_fp16_naive(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0);

// FP16 conversion utilities
void float_to_half(const float* in, __half* out, size_t n, cudaStream_t stream = 0);
void half_to_float(const __half* in, float* out, size_t n, cudaStream_t stream = 0);

void cuda_maxpool2d(const float* input, float* output, int* indices,
                   const Pool2dDesc& desc, cudaStream_t stream = 0);
void cuda_maxpool2d_backward(const float* grad_out, const float* input, const int* indices,
                              float* grad_in, const Pool2dDesc& desc, cudaStream_t stream = 0);

void cuda_dropout(float* data, float* mask, size_t size, float dropout_prob,
                  bool training, cudaStream_t stream = 0);
void cuda_dropout(const float* input, float* output, float* mask, size_t size,
                  float dropout_prob, bool training, cudaStream_t stream = 0);
void cuda_dropout_backward(const float* grad_out, const float* mask,
                           float* grad_in, size_t size, cudaStream_t stream = 0);

// CrossEntropyLoss (with C export for Python binding)
extern "C" {
void cuda_cross_entropy_loss(const float* logits, const int* targets,
                              float* loss, float* grad_logits,
                              size_t batch_size, size_t num_classes,
                              cudaStream_t stream = 0);

void cuda_sgd_update(float* param, const float* grad,
                      size_t size, float learning_rate,
                      cudaStream_t stream = 0);

void cuda_flatten(const float* input, float* output,
                   size_t batch, size_t C, size_t H, size_t W,
                   cudaStream_t stream = 0);

void cuda_flatten_backward(const float* grad_flat, float* grad_input,
                            size_t batch, size_t C, size_t H, size_t W,
                            cudaStream_t stream = 0);
}
