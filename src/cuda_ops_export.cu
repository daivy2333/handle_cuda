#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
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

extern "C" {

// Memory allocation
void* cuda_alloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_alloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

// Memory free
void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// Host to device copy
void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

// Device to host copy
void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

// Device synchronization
void cuda_sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Memory set
void cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

// Allocate and copy host to device
void* cuda_alloc_and_copy(const void* host_ptr, size_t size) {
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_alloc_and_copy failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));
    return dev_ptr;
}

// ============== MatMul C API ==============
// MatMul forward: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
void cuda_matmul_f32(const float* A, const float* B, float* C,
                     size_t M, size_t N, size_t K) {
    MatMulDesc desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.transpose_a = false;
    desc.transpose_b = false;
    cuda_matmul(A, B, C, desc, 0);
}

// MatMul backward
// grad_A = grad_C @ B^T: [M, K]
// grad_B = A^T @ grad_C: [K, N]
void cuda_matmul_backward_f32(const float* grad_C, const float* A, const float* B,
                               float* grad_A, float* grad_B,
                               size_t M, size_t N, size_t K) {
    MatMulDesc desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.transpose_a = false;
    desc.transpose_b = false;
    cuda_matmul_backward(grad_C, A, B, grad_A, grad_B, desc, 0);
}

// ============== BiasAdd C API ==============
// BiasAdd forward: output = input + bias (broadcast)
// input: [rows, cols], bias: [cols]
void cuda_bias_add_f32(const float* input, const float* bias, float* output,
                       size_t rows, size_t cols) {
    cuda_bias_add(input, bias, output, rows, cols, 0);
}

// BiasAdd backward
// grad_input = grad_out (copy), grad_bias = sum(grad_out over rows)
void cuda_bias_add_backward_f32(const float* grad_out, float* grad_input,
                                 float* grad_bias, size_t rows, size_t cols) {
    cuda_bias_add_backward(grad_out, grad_input, grad_bias, rows, cols, 0);
}

// ============== ReLU C API ==============
// ReLU forward: inplace activation
// data: [size]
void cuda_relu_f32(float* data, size_t size) {
    cuda_relu(data, size, 0);
}

// ReLU backward
// grad_in = grad_out * (forward_input > 0)
void cuda_relu_backward_f32(const float* grad_out, const float* forward_input,
                             float* grad_in, size_t size) {
    cuda_relu_backward(grad_out, forward_input, grad_in, size, 0);
}

// ============== Softmax C API ==============
// Softmax forward: output = softmax(input)
// input/output: [batch, classes]
void cuda_softmax_f32(const float* input, float* output,
                      size_t batch, size_t classes) {
    cuda_softmax(input, output, batch, classes, 0);
}

// Softmax backward
void cuda_softmax_backward_f32(const float* grad_out, const float* forward_output,
                                float* grad_in, size_t batch, size_t classes) {
    cuda_softmax_backward(grad_out, forward_output, grad_in, batch, classes, 0);
}

// ============== Conv2d C API ==============
// Winograd F(4x4, 3x3) conv2d forward
void cuda_conv2d_winograd_f32(const float* input, const float* weight, const float* bias,
                              float* output, float* temp_buffer,
                              int N, int C, int H, int W, int out_C,
                              int stride_h, int stride_w, int pad_h, int pad_w) {
    cuda_conv2d_winograd_forward(input, weight, bias, output, temp_buffer,
                                 N, C, H, W, out_C,
                                 stride_h, stride_w, pad_h, pad_w, 0);
}

// Conv2d forward: input [N, C, H, W], weight [out_C, C, kernel_h, kernel_w], output [N, out_C, out_H, out_W]
void cuda_conv2d_f32(const float* input, const float* weight, const float* bias, float* output,
                     int N, int C, int H, int W,
                     int out_C, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int pad_h, int pad_w) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;  // Standard convolution (groups=1)

    // Calculate output dimensions
    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    cuda_conv2d(input, weight, bias, output, desc, 0);
}

// Conv2d im2col: im2col + GEMM approach (much faster than naive)
void cuda_conv2d_im2col_f32(const float* input, const float* weight, const float* bias,
                             float* output, float* col_buffer, float* gemm_buffer,
                             int N, int C, int H, int W,
                             int out_C, int kernel_h, int kernel_w,
                             int stride_h, int stride_w, int pad_h, int pad_w) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;

    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    cuda_conv2d_im2col(input, weight, bias, output, col_buffer, gemm_buffer, desc, 0);
}

// Conv2d backward: compute grad_input, grad_weight, grad_bias
void cuda_conv2d_backward_f32(const float* grad_out, const float* input, const float* weight,
                              float* grad_input, float* grad_weight, float* grad_bias,
                              int N, int C, int H, int W,
                              int out_C, int kernel_h, int kernel_w,
                              int stride_h, int stride_w, int pad_h, int pad_w) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;

    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    cuda_conv2d_backward(grad_out, input, weight, grad_input, grad_weight, grad_bias, desc, 0);
}

// Conv2d backward with pre-allocated buffers (optimized, no malloc/free per batch)
void cuda_conv2d_backward_with_buffers_f32(const float* grad_out, const float* input, const float* weight,
                                            float* grad_input, float* grad_weight, float* grad_bias,
                                            float* reshaped_grad, float* col_buffer, float* col_grad,
                                            int N, int C, int H, int W,
                                            int out_C, int kernel_h, int kernel_w,
                                            int stride_h, int stride_w, int pad_h, int pad_w) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;

    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    cuda_conv2d_backward_with_buffers(grad_out, input, weight, grad_input, grad_weight, grad_bias,
                                       desc, reshaped_grad, col_buffer, col_grad, 0);
}

// Calculate conv2d backward buffer sizes
// Returns total size, fills reshaped_size and col_size
size_t cuda_conv2d_backward_buffer_sizes_f32(int N, int C, int H, int W,
                                               int out_C, int kernel_h, int kernel_w,
                                               int stride_h, int stride_w, int pad_h, int pad_w,
                                               size_t* reshaped_size, size_t* col_size) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;

    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    return cuda_conv2d_backward_buffer_sizes(desc, reshaped_size, col_size);
}

// ============== MaxPool2d C API ==============
// MaxPool2d forward: input [N, C, H, W] -> output [N, C, out_H, out_W]
// indices stores the index of max element for backward pass
void cuda_maxpool2d_f32(const float* input, float* output, int* indices,
                        int N, int C, int H, int W,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w, int pad_h, int pad_w) {
    Pool2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;

    cuda_maxpool2d(input, output, indices, desc, 0);
}

// MaxPool2d backward: scatter grad_out to grad_input using indices
void cuda_maxpool2d_backward_f32(const float* grad_out, const int* indices, float* grad_input,
                                  int N, int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int stride_h, int stride_w, int pad_h, int pad_w) {
    Pool2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;

    cuda_maxpool2d_backward(grad_out, nullptr, indices, grad_input, desc, 0);
}

} // extern "C"