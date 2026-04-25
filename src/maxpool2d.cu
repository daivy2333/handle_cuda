#include "cuda_ops.h"
#include "cuda_util.h"
#include <cmath>

namespace {

__global__ void maxpool2d_kernel(const float* input, float* output, int* indices,
                                   int N, int C, int H, int W,
                                   int out_H, int out_W,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x / out_W;
    int ow = blockIdx.x % out_W;

    if (n < N && c < C && oh < out_H && ow < out_W) {
        float max_val = -INFINITY;
        int max_idx = -1;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h + kh - pad_h;
                int iw = ow * stride_w + kw - pad_w;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int idx = n * C * H * W + c * H * W + ih * W + iw;
                    if (input[idx] > max_val) {
                        max_val = input[idx];
                        max_idx = idx;
                    }
                }
            }
        }

        int out_idx = n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
        output[out_idx] = max_val;
        indices[out_idx] = max_idx;
    }
}

__global__ void maxpool2d_backward_kernel(const float* grad_out, const float* input,
                                            const int* indices, float* grad_in,
                                            int N, int C, int H, int W,
                                            int out_H, int out_W,
                                            int kernel_h, int kernel_w,
                                            int stride_h, int stride_w,
                                            int pad_h, int pad_w) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x / out_W;
    int ow = blockIdx.x % out_W;

    if (n < N && c < C && oh < out_H && ow < out_W) {
        int out_idx = n * C * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
        int max_idx = indices[out_idx];

        if (max_idx >= 0 && max_idx < N * C * H * W) {
            atomicAdd(&grad_in[max_idx], grad_out[out_idx]);
        }
    }
}

} // namespace

void cuda_maxpool2d(const float* input, float* output, int* indices,
                     const Pool2dDesc& desc, cudaStream_t stream) {
    int out_W = desc.W / desc.stride_w;
    int total_h = desc.H / desc.stride_h;

    dim3 block_dim(16, 16);
    dim3 grid_dim(total_h * out_W, desc.C, desc.N);

    maxpool2d_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, output, indices,
        desc.N, desc.C, desc.H, desc.W,
        desc.H / desc.stride_h, desc.W / desc.stride_w,
        desc.kernel_h, desc.kernel_w,
        desc.stride_h, desc.stride_w,
        desc.pad_h, desc.pad_w);

    CUDA_CHECK(cudaGetLastError());
}

void cuda_maxpool2d_backward(const float* grad_out, const float* input, const int* indices,
                               float* grad_in, const Pool2dDesc& desc, cudaStream_t stream) {
    int out_W = desc.W / desc.stride_w;
    int out_H = desc.H / desc.stride_h;
    int total_h = out_H;

    dim3 block_dim(16, 16);
    dim3 grid_dim(total_h * out_W, desc.C, desc.N);

    maxpool2d_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        grad_out, input, indices, grad_in,
        desc.N, desc.C, desc.H, desc.W,
        out_H, out_W,
        desc.kernel_h, desc.kernel_w,
        desc.stride_h, desc.stride_w,
        desc.pad_h, desc.pad_w);

    CUDA_CHECK(cudaGetLastError());
}
