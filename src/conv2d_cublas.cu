// src/conv2d_cublas.cu - Conv2d using im2col + cuBLAS sgemm
#include "cuda_ops.h"
#include "cuda_util.h"
#include <cublas_v2.h>

namespace {

// Static cuBLAS handle
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = []{
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

// im2col kernel (same as before)
__global__ void im2col_kernel(const float* input, float* col,
                              int N, int C, int H, int W,
                              int out_H, int out_W,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;

    if (idx < col_rows * col_cols) {
        int col_row = idx / col_cols;
        int col_col = idx % col_cols;

        int c = col_row / (kernel_h * kernel_w);
        int kh = (col_row % (kernel_h * kernel_w)) / kernel_w;
        int kw = col_row % kernel_w;

        int n = col_col / (out_H * out_W);
        int oh = (col_col % (out_H * out_W)) / out_W;
        int ow = col_col % out_W;

        int ih = oh * stride_h + kh - pad_h;
        int iw = ow * stride_w + kw - pad_w;

        col[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ?
                   input[n * C * H * W + c * H * W + ih * W + iw] : 0.0f;
    }
}

// col2im kernel for backward
__global__ void col2im_kernel(const float* col, float* input,
                              int N, int C, int H, int W,
                              int out_H, int out_W,
                              int kernel_h, int kernel_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;

    if (idx < col_rows * col_cols) {
        int col_row = idx / col_cols;
        int col_col = idx % col_cols;

        int c = col_row / (kernel_h * kernel_w);
        int kh = (col_row % (kernel_h * kernel_w)) / kernel_w;
        int kw = col_row % kernel_w;

        int n = col_col / (out_H * out_W);
        int oh = (col_col % (out_H * out_W)) / out_W;
        int ow = col_col % out_W;

        int ih = oh * stride_h + kh - pad_h;
        int iw = ow * stride_w + kw - pad_w;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            // Atomic add because multiple col entries map to same input
            atomicAdd(&input[n * C * H * W + c * H * W + ih * W + iw], col[idx]);
        }
    }
}

// Bias add kernel for conv output [N, out_C, out_H, out_W] layout
__global__ void bias_add_output_kernel(const float* data, const float* bias,
                                       float* output, int N, int out_C, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int oc = (idx / (out_H * out_W)) % out_C;
        output[idx] = data[idx] + bias[oc];
    }
}

// Reshape output kernel (transpose from GEMM output)
__global__ void reshape_output_kernel(const float* gemm_out, float* output,
                                       int out_C, int N, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int n = idx / (out_C * out_H * out_W);
        int remaining = idx % (out_C * out_H * out_W);
        int oc = remaining / (out_H * out_W);
        int spatial = remaining % (out_H * out_W);
        int oh = spatial / out_W;
        int ow = spatial % out_W;

        // GEMM output is [out_C, N*out_H*out_W]
        int gemm_idx = oc * (N * out_H * out_W) + n * (out_H * out_W) + oh * out_W + ow;
        output[idx] = gemm_out[gemm_idx];
    }
}

// Reshape grad_output for backward GEMM
__global__ void reshape_grad_kernel(const float* grad_out, float* grad_gemm,
                                    int out_C, int N, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int n = idx / (out_C * out_H * out_W);
        int remaining = idx % (out_C * out_H * out_W);
        int oc = remaining / (out_H * out_W);
        int spatial = remaining % (out_H * out_W);
        int oh = spatial / out_W;
        int ow = spatial % out_W;

        // grad_gemm is [out_C, N*out_H*out_W]
        int gemm_idx = oc * (N * out_H * out_W) + n * (out_H * out_W) + oh * out_W + ow;
        grad_gemm[gemm_idx] = grad_out[idx];
    }
}

// Kernel to zero out input gradient
__global__ void zero_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

// Bias backward kernel: sum grad_output over batch and spatial dimensions
__global__ void bias_backward_kernel(const float* grad_output, float* grad_bias,
                                      int N, int out_C, int out_H, int out_W) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc < out_C) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    int idx = n * out_C * out_H * out_W + oc * out_H * out_W + oh * out_W + ow;
                    sum += grad_output[idx];
                }
            }
        }
        grad_bias[oc] = sum;
    }
}

} // namespace

// Forward: im2col + cuBLAS sgemm
void cuda_conv2d_im2col_cublas(
    const float* input, const float* weight, const float* bias,
    float* output, float* col_buffer, float* gemm_buffer,
    int N, int C, int H, int W, int out_C,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    int out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;

    // Step 1: im2col
    int col_size = col_rows * col_cols;
    int block_size = 256;
    int num_blocks = (col_size + block_size - 1) / block_size;

    im2col_kernel<<<num_blocks, block_size>>>(input, col_buffer,
        N, C, H, W, out_H, out_W,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: cuBLAS sgemm using existing matmul_cublas
    // We want: gemm_buffer[out_C][col_cols] = weight[out_C][col_rows] @ col_buffer[col_rows][col_cols]
    // Use cuda_matmul_cublas which handles row-major correctly

    extern void cuda_matmul_cublas(const float* A, const float* B, float* C,
                                   size_t M, size_t N, size_t K, cudaStream_t stream);
    cuda_matmul_cublas(weight, col_buffer, gemm_buffer, out_C, col_cols, col_rows, 0);

    CUDA_CHECK(cudaGetLastError());

    // Step 3: Reshape output [out_C, N*out_H*out_W] -> [N, out_C, out_H, out_W]
    int output_size = N * out_C * out_H * out_W;
    num_blocks = (output_size + block_size - 1) / block_size;
    reshape_output_kernel<<<num_blocks, block_size>>>(gemm_buffer, output, out_C, N, out_H, out_W);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: Add bias to reshaped output (if provided)
    if (bias) {
        bias_add_output_kernel<<<num_blocks, block_size>>>(output, bias, output, N, out_C, out_H, out_W);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Backward: grad_weight, grad_bias, grad_input using cuBLAS
void cuda_conv2d_im2col_cublas_backward(
    const float* grad_output, const float* input, const float* weight,
    float* grad_input, float* grad_weight, float* grad_bias,
    float* col_buffer, float* grad_col_buffer, float* grad_gemm_buffer,
    int N, int C, int H, int W, int out_C,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    int out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;

    int block_size = 256;
    int num_blocks;

    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;

    // Step 1: Reshape grad_output [N, out_C, out_H, out_W] -> [out_C, N*out_H*out_W]
    int output_size = N * out_C * out_H * out_W;
    num_blocks = (output_size + block_size - 1) / block_size;
    reshape_grad_kernel<<<num_blocks, block_size>>>(grad_output, grad_gemm_buffer, out_C, N, out_H, out_W);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: grad_bias = sum over N*out_H*out_W
    // Use reshape kernel result: grad_gemm_buffer[out_C][col_cols]
    // Sum each row of grad_gemm_buffer
    if (grad_bias) {
        // Sum grad_output over all spatial dimensions
        int num_blocks_bias = (out_C + block_size - 1) / block_size;
        bias_backward_kernel<<<num_blocks_bias, block_size>>>(grad_output, grad_bias, N, out_C, out_H, out_W);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: grad_weight = grad_gemm @ col_buffer^T
    // grad_gemm row-major [out_C][col_cols], col_buffer row-major [col_rows][col_cols]
    // grad_weight row-major [out_C][col_rows] = grad_gemm @ col_buffer^T
    //
    // This is C = A @ B^T where A=grad_gemm, B=col_buffer
    // Use cuda_matmul with transpose_b=true

    // First compute im2col for forward input (need for backward)
    num_blocks = (col_rows * col_cols + block_size - 1) / block_size;
    im2col_kernel<<<num_blocks, block_size>>>(input, col_buffer,
        N, C, H, W, out_H, out_W,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    CUDA_CHECK(cudaGetLastError());

    if (grad_weight) {
        extern void cuda_matmul(const float* A, const float* B, float* C,
                                const MatMulDesc& desc, cudaStream_t stream);
        MatMulDesc desc_weight{
            static_cast<size_t>(out_C),
            static_cast<size_t>(col_rows),
            static_cast<size_t>(col_cols),
            false, true  // transpose_b = true for col_buffer^T
        };
        cuda_matmul(grad_gemm_buffer, col_buffer, grad_weight, desc_weight, 0);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: grad_input via col2im
    // grad_col = weight^T @ grad_gemm
    // weight row-major [out_C][col_rows], grad_gemm row-major [out_C][col_cols]
    // grad_col row-major [col_rows][col_cols] = weight^T @ grad_gemm
    //
    // This is C = A^T @ B where A=weight, B=grad_gemm
    // Use cuda_matmul with transpose_a=true

    if (grad_input) {
        // Zero grad_input first
        int input_size = N * C * H * W;
        num_blocks = (input_size + block_size - 1) / block_size;
        zero_kernel<<<num_blocks, block_size>>>(grad_input, input_size);
        CUDA_CHECK(cudaGetLastError());

        // grad_col = weight^T @ grad_gemm
        extern void cuda_matmul(const float* A, const float* B, float* C,
                                const MatMulDesc& desc, cudaStream_t stream);
        MatMulDesc desc_col{
            static_cast<size_t>(col_rows),
            static_cast<size_t>(col_cols),
            static_cast<size_t>(out_C),
            true, false  // transpose_a = true for weight^T
        };
        cuda_matmul(weight, grad_gemm_buffer, grad_col_buffer, desc_col, 0);
        CUDA_CHECK(cudaGetLastError());

        // col2im
        num_blocks = (col_rows * col_cols + block_size - 1) / block_size;
        col2im_kernel<<<num_blocks, block_size>>>(grad_col_buffer, grad_input,
            N, C, H, W, out_H, out_W,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
        CUDA_CHECK(cudaGetLastError());
    }
}