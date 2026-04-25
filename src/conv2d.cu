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
