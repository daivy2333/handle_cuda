#include "cuda_ops.h"
#include "cuda_util.h"
#include <cstdio>

namespace {

// Simple im2col + GEMM conv2d - for comparison with Winograd
// This is essentially what we already have in conv2d.cu, but simplified for reference

__global__ void im2col_kernel_simple(
    const float* input, float* col,
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

__global__ void bias_add_kernel_simple(const float* data, const float* bias,
                                        float* output, int N, int out_C, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int oc = (idx / (out_H * out_W)) % out_C;
        output[idx] = data[idx] + bias[oc];
    }
}

__global__ void reshape_output_kernel_simple(const float* gemm_output, float* output,
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

        int gemm_idx = oc * (N * out_H * out_W) + n * (out_H * out_W) + oh * out_W + ow;
        output[idx] = gemm_output[gemm_idx];
    }
}

} // namespace

// Wrapper that mimics Winograd's interface for comparison
void cuda_conv2d_simple_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_buffer,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream) {
    
    printf("[SimpleConv] Starting: N=%d, C=%d, H=%d, W=%d, out_C=%d\n", 
           N, C, H, W, out_C);
    
    int out_H = (H + 2 * pad_h - 3) / stride_h + 1;
    int out_W = (W + 2 * pad_w - 3) / stride_w + 1;
    
    printf("[SimpleConv] out_H=%d, out_W=%d\n", out_H, out_W);
    
    // temp_buffer layout: col_buffer + gemm_buffer
    float* col_buffer = temp_buffer;
    float* gemm_buffer = temp_buffer + C * 3 * 3 * N * out_H * out_W;
    
    // im2col
    int col_rows = C * 3 * 3;
    int col_cols = N * out_H * out_W;
    int total_col = col_rows * col_cols;
    int block_size = 256;
    int num_blocks = (total_col + block_size - 1) / block_size;
    
    printf("[SimpleConv] Running im2col\n");
    im2col_kernel_simple<<<num_blocks, block_size, 0, stream>>>(
        input, col_buffer, N, C, H, W, out_H, out_W, 3, 3, pad_h, pad_w, stride_h, stride_w);
    CUDA_CHECK(cudaGetLastError());
    
    // GEMM: weight @ col_buffer
    MatMulDesc gemm_desc;
    gemm_desc.M = out_C;
    gemm_desc.N = col_cols;
    gemm_desc.K = col_rows;
    gemm_desc.transpose_a = false;
    gemm_desc.transpose_b = false;
    
    printf("[SimpleConv] Running matmul\n");
    cuda_matmul(weight, col_buffer, gemm_buffer, gemm_desc, stream);
    CUDA_CHECK(cudaGetLastError());
    
    // Reshape output
    int total_out = N * out_C * out_H * out_W;
    int reshape_blocks = (total_out + 255) / 256;
    
    printf("[SimpleConv] Reshaping output\n");
    reshape_output_kernel_simple<<<reshape_blocks, 256, 0, stream>>>(
        gemm_buffer, output, out_C, N, out_H, out_W);
    
    // Add bias
    if (bias != nullptr) {
        printf("[SimpleConv] Adding bias\n");
        int bias_blocks = (total_out + 255) / 256;
        bias_add_kernel_simple<<<bias_blocks, 256, 0, stream>>>(
            output, bias, output, N, out_C, out_H, out_W);
    }
    
    printf("[SimpleConv] Complete\n");
    CUDA_CHECK(cudaGetLastError());
}
