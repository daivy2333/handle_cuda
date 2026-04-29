#include "cuda_ops.h"
#include "cuda_util.h"
#include <cmath>

namespace {

// ============================================================================
// Forward kernels
// ============================================================================

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

__global__ void bias_add_output_kernel(const float* data, const float* bias,
                                       float* output, int N, int out_C, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int oc = (idx / (out_H * out_W)) % out_C;
        output[idx] = data[idx] + bias[oc];
    }
}

__global__ void reshape_output_kernel(const float* gemm_output, float* output,
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

__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias,
                               float* output, int N, int C, int H, int W,
                               int out_C, int out_H, int out_W,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w, int groups) {
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x / out_W;
    int ow = blockIdx.x % out_W;

    if (n < N && oc < out_C && oh < out_H && ow < out_W) {
        int out_c_per_group = out_C / groups;
        int in_c_per_group = C / groups;
        int group_idx = oc / out_c_per_group;
        int in_c_start = group_idx * in_c_per_group;
        int in_c_end = in_c_start + in_c_per_group;

        float sum = 0.0f;
        for (int inc = in_c_start; inc < in_c_end; ++inc) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int ih = oh * stride_h + kh - pad_h;
                    int iw = ow * stride_w + kw - pad_w;

                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        int input_idx = n * C * H * W + inc * H * W + ih * W + iw;
                        int weight_idx = oc * in_c_per_group * kernel_h * kernel_w +
                                        (inc - in_c_start) * kernel_h * kernel_w +
                                        kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[n * out_C * out_H * out_W + oc * out_H * out_W + oh * out_W + ow] = sum;
    }
}

// ============================================================================
// Backward kernels
// ============================================================================

// Parallel reduction for bias gradient (optimized)
__global__ void conv2d_backward_bias_kernel_opt(const float* grad_out, float* grad_bias,
                                                  int total_spatial, int out_C, int spatial_offset) {
    int oc = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared_sum[256];

    float sum = 0.0f;
    for (int i = tid; i < total_spatial; i += blockDim.x) {
        sum += grad_out[oc * spatial_offset + i];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[oc] = shared_sum[0];
    }
}

// Reshape grad_out [N, out_C, out_H, out_W] -> [out_C, N*out_H*out_W]
__global__ void reshape_grad_for_backward_kernel(const float* grad_out, float* reshaped,
                                                  int N, int out_C, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int n = idx / (out_C * out_H * out_W);
        int remaining = idx % (out_C * out_H * out_W);
        int oc = remaining / (out_H * out_W);
        int spatial = remaining % (out_H * out_W);
        int oh = spatial / out_W;
        int ow = spatial % out_W;

        int reshaped_idx = oc * (N * out_H * out_W) + n * (out_H * out_W) + oh * out_W + ow;
        reshaped[reshaped_idx] = grad_out[idx];
    }
}

// col2im - scatter gradient back to input shape
__global__ void col2im_backward_kernel(const float* col_grad, float* input_grad,
                                        int N, int C, int H, int W,
                                        int out_H, int out_W,
                                        int kernel_h, int kernel_w,
                                        int pad_h, int pad_w,
                                        int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_rows = C * kernel_h * kernel_w;
    int col_cols = N * out_H * out_W;

    if (idx < col_rows * col_cols) {
        float val = col_grad[idx];

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
            atomicAdd(&input_grad[n * C * H * W + c * H * W + ih * W + iw], val);
        }
    }
}

// Naive fallback kernels for grad_weight and non-optimized cases
__global__ void conv2d_backward_weight_kernel(const float* grad_out, const float* input,
                                               float* grad_weight,
                                               int N, int C, int H, int W,
                                               int out_C, int out_H, int out_W,
                                               int kernel_h, int kernel_w,
                                               int stride_h, int stride_w,
                                               int pad_h, int pad_w) {
    int oc = blockIdx.z;
    int c = blockIdx.y;
    int kh = blockIdx.x / kernel_w;
    int kw = blockIdx.x % kernel_w;

    if (oc < out_C && c < C && kh < kernel_h && kw < kernel_w) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    int ih = oh * stride_h + kh - pad_h;
                    int iw = ow * stride_w + kw - pad_w;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        sum += grad_out[n * out_C * out_H * out_W +
                                       oc * out_H * out_W + oh * out_W + ow] *
                              input[n * C * H * W + c * H * W + ih * W + iw];
                    }
                }
            }
        }
        grad_weight[oc * C * kernel_h * kernel_w +
                   c * kernel_h * kernel_w + kh * kernel_w + kw] = sum;
    }
}

__global__ void conv2d_backward_input_kernel_naive(const float* grad_out, const float* weight,
                                                    float* grad_input,
                                                    int N, int C, int H, int W,
                                                    int out_C, int out_H, int out_W,
                                                    int kernel_h, int kernel_w,
                                                    int stride_h, int stride_w,
                                                    int pad_h, int pad_w) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int ih = blockIdx.x / W;
    int iw = blockIdx.x % W;

    if (n < N && c < C && ih < H && iw < W) {
        float sum = 0.0f;
        for (int oc = 0; oc < out_C; ++oc) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int oh = ih + pad_h - kh;
                    int ow = iw + pad_w - kw;
                    if (oh % stride_h == 0 && ow % stride_w == 0 &&
                        oh >= 0 && oh < out_H && ow >= 0 && ow < out_W) {
                        int oh_idx = oh / stride_h;
                        int ow_idx = ow / stride_w;
                        sum += grad_out[n * out_C * out_H * out_W +
                                       oc * out_H * out_W + oh_idx * out_W + ow_idx] *
                              weight[oc * C * kernel_h * kernel_w +
                                    c * kernel_h * kernel_w + kh * kernel_w + kw];
                    }
                }
            }
        }
        grad_input[n * C * H * W + c * H * W + ih * W + iw] = sum;
    }
}

} // namespace

// ============================================================================
// Forward functions
// ============================================================================

void cuda_conv2d(const float* input, const float* weight, const float* bias, float* output,
                 const Conv2dDesc& desc, cudaStream_t stream) {
    dim3 block_dim(16, 16);
    dim3 grid_dim(desc.out_H * desc.out_W, desc.out_C, desc.N);

    conv2d_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, weight, bias, output,
        desc.N, desc.C, desc.H, desc.W,
        desc.out_C, desc.out_H, desc.out_W,
        desc.kernel_h, desc.kernel_w,
        desc.stride_h, desc.stride_w,
        desc.pad_h, desc.pad_w, desc.groups);

    CUDA_CHECK(cudaGetLastError());
}

void cuda_conv2d_im2col(const float* input, const float* weight, const float* bias,
                        float* output, float* col_buffer, float* gemm_buffer,
                        const Conv2dDesc& desc, cudaStream_t stream) {
    int col_rows = desc.C * desc.kernel_h * desc.kernel_w;
    int col_cols = desc.N * desc.out_H * desc.out_W;

    int total_col = col_rows * col_cols;
    int block_size = 256;
    int num_blocks = (total_col + block_size - 1) / block_size;

    im2col_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, col_buffer,
        desc.N, desc.C, desc.H, desc.W,
        desc.out_H, desc.out_W,
        desc.kernel_h, desc.kernel_w,
        desc.pad_h, desc.pad_w,
        desc.stride_h, desc.stride_w);

    MatMulDesc gemm_desc{
        static_cast<size_t>(desc.out_C),
        static_cast<size_t>(col_cols),
        static_cast<size_t>(col_rows),
        false, false
    };

    cuda_matmul(weight, col_buffer, gemm_buffer, gemm_desc, stream);

    int total_out = desc.N * desc.out_C * desc.out_H * desc.out_W;
    int reshape_blocks = (total_out + 255) / 256;
    reshape_output_kernel<<<reshape_blocks, 256, 0, stream>>>(
        gemm_buffer, output, desc.out_C, desc.N, desc.out_H, desc.out_W);

    if (bias != nullptr) {
        bias_add_output_kernel<<<reshape_blocks, 256, 0, stream>>>(
            output, bias, output, desc.N, desc.out_C, desc.out_H, desc.out_W);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Backward function - OPTIMIZED using im2col + GEMM
// ============================================================================

void cuda_conv2d_backward(const float* grad_out, const float* input, const float* weight,
                          float* grad_input, float* grad_weight, float* grad_bias,
                          const Conv2dDesc& desc, cudaStream_t stream) {
    int N = desc.N;
    int C = desc.C;
    int H = desc.H;
    int W = desc.W;
    int out_C = desc.out_C;
    int out_H = desc.out_H;
    int out_W = desc.out_W;
    int K = desc.kernel_h;

    // ===== Bias gradient - optimized with parallel reduction =====
    int total_spatial = N * out_H * out_W;
    int spatial_offset = out_H * out_W;
    conv2d_backward_bias_kernel_opt<<<out_C, 256, 0, stream>>>(
        grad_out, grad_bias, total_spatial, out_C, spatial_offset);

    // ===== Check if we can use im2col optimization =====
    bool can_optimize = (desc.stride_h == 1 && desc.stride_w == 1 &&
                         desc.groups == 1 && desc.kernel_h == desc.kernel_w);

    if (can_optimize) {
        // ===== Optimized path: im2col + GEMM + col2im =====
        int col_rows = C * K * K;
        int col_cols = N * out_H * out_W;

        // Allocate temporary buffers
        float* reshaped_grad = nullptr;
        float* col_grad = nullptr;

        size_t reshaped_size = out_C * col_cols * sizeof(float);
        size_t col_grad_size = col_rows * col_cols * sizeof(float);

        cudaMalloc(&reshaped_grad, reshaped_size);
        cudaMalloc(&col_grad, col_grad_size);

        // Step 1: Reshape grad_out [N, out_C, out_H, out_W] -> [out_C, col_cols]
        int total_grad = N * out_C * out_H * out_W;
        int reshape_blocks = (total_grad + 255) / 256;
        reshape_grad_for_backward_kernel<<<reshape_blocks, 256, 0, stream>>>(
            grad_out, reshaped_grad, N, out_C, out_H, out_W);

        // Step 2: grad_weight = reshaped_grad @ im2col(input)^T
        // First compute im2col(input)
        float* col_buffer = nullptr;
        size_t col_buffer_size = col_rows * col_cols * sizeof(float);
        cudaMalloc(&col_buffer, col_buffer_size);

        int total_col = col_rows * col_cols;
        int col_blocks = (total_col + 255) / 256;
        im2col_kernel<<<col_blocks, 256, 0, stream>>>(
            input, col_buffer, N, C, H, W, out_H, out_W, K, K,
            desc.pad_h, desc.pad_w, 1, 1);

        // grad_weight = reshaped_grad @ col_buffer^T
        // reshaped_grad: [out_C, col_cols]
        // col_buffer: [col_rows, col_cols]
        // grad_weight: [out_C, col_rows]
        MatMulDesc weight_grad_desc;
        weight_grad_desc.M = out_C;          // output rows
        weight_grad_desc.N = col_rows;       // output cols
        weight_grad_desc.K = col_cols;       // reduction dim
        weight_grad_desc.transpose_a = false;
        weight_grad_desc.transpose_b = true; // transpose col_buffer

        cuda_matmul(reshaped_grad, col_buffer, grad_weight, weight_grad_desc, stream);

        cudaFree(col_buffer);

        // Step 3: col_grad = weight^T @ reshaped_grad
        // weight: [out_C, col_rows]
        // reshaped_grad: [out_C, col_cols]
        // col_grad: [col_rows, col_cols]
        MatMulDesc matmul_desc;
        matmul_desc.M = col_rows;       // output rows
        matmul_desc.N = col_cols;       // output cols
        matmul_desc.K = out_C;          // reduction dim
        matmul_desc.transpose_a = true; // transpose weight
        matmul_desc.transpose_b = false;

        cuda_matmul(weight, reshaped_grad, col_grad, matmul_desc, stream);

        // Step 4: grad_input = col2im(col_grad)
        // Zero out grad_input first (col2im uses atomicAdd)
        cudaMemsetAsync(grad_input, 0, N * C * H * W * sizeof(float), stream);

        int col_total = col_rows * col_cols;
        int col2im_blocks = (col_total + 255) / 256;
        col2im_backward_kernel<<<col2im_blocks, 256, 0, stream>>>(
            col_grad, grad_input, N, C, H, W, out_H, out_W, K, K,
            desc.pad_h, desc.pad_w, 1, 1);

        // Free temporary buffers
        cudaFree(reshaped_grad);
        cudaFree(col_grad);
    } else {
        // ===== Fallback to naive kernels =====
        dim3 grid_input(H * W, C, N);
        dim3 block_input(16, 16);
        conv2d_backward_input_kernel_naive<<<grid_input, block_input, 0, stream>>>(
            grad_out, weight, grad_input, N, C, H, W, out_C, out_H, out_W,
            K, K, desc.stride_h, desc.stride_w, desc.pad_h, desc.pad_w);

        dim3 grid_weight(K * K, C, out_C);
        dim3 block_weight(16, 16);
        conv2d_backward_weight_kernel<<<grid_weight, block_weight, 0, stream>>>(
            grad_out, input, grad_weight, N, C, H, W, out_C, out_H, out_W,
            K, K, desc.stride_h, desc.stride_w, desc.pad_h, desc.pad_w);
    }

    CUDA_CHECK(cudaGetLastError());
}