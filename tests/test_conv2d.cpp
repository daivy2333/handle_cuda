#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>

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
