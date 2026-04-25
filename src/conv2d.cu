#include "cuda_ops.h"
#include "cuda_util.h"
#include <cmath>

namespace {

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
        int in_c_start = (oc * C) / out_C;
        int in_c_end = ((oc + 1) * C) / out_C;

        float sum = 0.0f;
        for (int inc = in_c_start; inc < in_c_end; ++inc) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int ih = oh * stride_h + kh - pad_h;
                    int iw = ow * stride_w + kw - pad_w;

                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        int input_idx = n * C * H * W + inc * H * W + ih * W + iw;
                        int weight_idx = oc * C / groups * kernel_h * kernel_w +
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

        int out_idx = n * out_C * out_H * out_W + oc * out_H * out_W + oh * out_W + ow;
        output[out_idx] = sum;
    }
}

} // namespace

__global__ void conv2d_backward_bias_kernel(const float* grad_out, float* grad_bias,
                                            int N, int out_C, int out_H, int out_W) {
    int oc = blockIdx.x;
    if (oc < out_C) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    sum += grad_out[n * out_C * out_H * out_W +
                                   oc * out_H * out_W + oh * out_W + ow];
                }
            }
        }
        grad_bias[oc] = sum;
    }
}

__global__ void conv2d_backward_input_kernel(const float* grad_out, const float* weight,
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
                    if (oh >= 0 && oh < out_H && ow >= 0 && ow < out_W &&
                        oh % stride_h == 0 && ow % stride_w == 0) {
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

void cuda_conv2d_backward(const float* grad_out, const float* input, const float* weight,
                          float* grad_input, float* grad_weight, float* grad_bias,
                          const Conv2dDesc& desc, cudaStream_t stream) {
    // grad_bias
    conv2d_backward_bias_kernel<<<desc.out_C, 1, 0, stream>>>(
        grad_out, grad_bias, desc.N, desc.out_C, desc.out_H, desc.out_W);

    // grad_input
    dim3 grid_input(desc.H * desc.W, desc.C, desc.N);
    dim3 block_input(16, 16);
    conv2d_backward_input_kernel<<<grid_input, block_input, 0, stream>>>(
        grad_out, weight, grad_input,
        desc.N, desc.C, desc.H, desc.W,
        desc.out_C, desc.out_H, desc.out_W,
        desc.kernel_h, desc.kernel_w,
        desc.stride_h, desc.stride_w,
        desc.pad_h, desc.pad_w);

    // grad_weight
    dim3 grid_weight(desc.kernel_h * desc.kernel_w, desc.C, desc.out_C);
    dim3 block_weight(16, 16);
    conv2d_backward_weight_kernel<<<grid_weight, block_weight, 0, stream>>>(
        grad_out, input, grad_weight,
        desc.N, desc.C, desc.H, desc.W,
        desc.out_C, desc.out_H, desc.out_W,
        desc.kernel_h, desc.kernel_w,
        desc.stride_h, desc.stride_w,
        desc.pad_h, desc.pad_w);

    CUDA_CHECK(cudaGetLastError());
}

void cuda_conv2d(const float* input, const float* weight, const float* bias, float* output,
                 const Conv2dDesc& desc, cudaStream_t stream) {
    int out_W = desc.out_W;
    int total_h = desc.out_H * desc.out_W;

    dim3 block_dim(16, 16);
    dim3 grid_dim(total_h, desc.out_C, desc.N);

    conv2d_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, weight, bias, output,
        desc.N, desc.C, desc.H, desc.W,
        desc.out_C, desc.out_H, desc.out_W,
        desc.kernel_h, desc.kernel_w,
        desc.stride_h, desc.stride_w,
        desc.pad_h, desc.pad_w, desc.groups);

    CUDA_CHECK(cudaGetLastError());
}
