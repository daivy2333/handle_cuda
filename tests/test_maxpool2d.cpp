#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>

class MaxPool2dTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    std::vector<float> generate_random(size_t size) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        return v;
    }
};

TEST_F(MaxPool2dTest, Basic) {
    Pool2dDesc desc;
    desc.N = 1; desc.C = 1; desc.H = 4; desc.W = 4;
    desc.kernel_h = 2; desc.kernel_w = 2;
    desc.stride_h = 2; desc.stride_w = 2;
    desc.pad_h = 0; desc.pad_w = 0;

    std::vector<float> input = {{
        1, 3, 2, 4,
        5, 9, 7, 8,
        6, 2, 1, 3,
        4, 7, 5, 6
    }};

    std::vector<float> output_ref = {{
        9, 8,
        7, 7
    }};

    size_t output_size = desc.N * desc.C * desc.H/2 * desc.W/2;
    CudaBuffer d_input(desc.N * desc.C * desc.H * desc.W),
               d_output(output_size);
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, output_size * sizeof(int)));

    host_to_device_async(d_input.data, input.data(), input.size());

    cuda_maxpool2d(d_input.data, d_output.data, d_indices, desc);

    std::vector<float> output(output_size);
    device_to_host(d_output.data, output.data(), output_size);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], output_ref[i]);
    }

    cudaFree(d_indices);
}

TEST_F(MaxPool2dTest, WithPadding) {
    Pool2dDesc desc;
    desc.N = 1; desc.C = 1; desc.H = 5; desc.W = 5;
    desc.kernel_h = 3; desc.kernel_w = 3;
    desc.stride_h = 2; desc.stride_w = 2;
    desc.pad_h = 1; desc.pad_w = 1;

    std::vector<float> input = {{
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    }};

    std::vector<float> output_ref = {{
        9, 10,
        14, 15,
        24, 25
    }};

    size_t output_size = desc.N * desc.C * 3 * 3;
    CudaBuffer d_input(desc.N * desc.C * desc.H * desc.W),
               d_output(output_size);
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, output_size * sizeof(int)));

    host_to_device_async(d_input.data, input.data(), input.size());

    cuda_maxpool2d(d_input.data, d_output.data, d_indices, desc);

    std::vector<float> output(output_size);
    device_to_host(d_output.data, output.data(), output_size);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], output_ref[i]);
    }

    cudaFree(d_indices);
}

TEST_F(MaxPool2dTest, BatchAndChannels) {
    Pool2dDesc desc;
    desc.N = 2; desc.C = 2; desc.H = 4; desc.W = 4;
    desc.kernel_h = 2; desc.kernel_w = 2;
    desc.stride_h = 2; desc.stride_w = 2;
    desc.pad_h = 0; desc.pad_w = 0;

    size_t input_size = desc.N * desc.C * desc.H * desc.W;
    size_t output_size = desc.N * desc.C * 2 * 2;
    auto input = generate_random(input_size);

    std::vector<float> output_ref(desc.N * desc.C * 2 * 2);

    for (int n = 0; n < desc.N; ++n) {
        for (int c = 0; c < desc.C; ++c) {
            for (int oh = 0; oh < 2; ++oh) {
                for (int ow = 0; ow < 2; ++ow) {
                    float max_val = -INFINITY;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int ih = oh * 2 + kh;
                            int iw = ow * 2 + kw;
                            int idx = n * desc.C * desc.H * desc.W + c * desc.H * desc.W + ih * desc.W + iw;
                            max_val = std::fmax(max_val, input[idx]);
                        }
                    }
                    int out_idx = n * desc.C * 2 * 2 + c * 2 * 2 + oh * 2 + ow;
                    output_ref[out_idx] = max_val;
                }
            }
        }
    }

    CudaBuffer d_input(input_size),
               d_output(output_size);
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, output_size * sizeof(int)));

    host_to_device_async(d_input.data, input.data(), input_size);

    cuda_maxpool2d(d_input.data, d_output.data, d_indices, desc);

    std::vector<float> output(output_size);
    device_to_host(d_output.data, output.data(), output_size);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], output_ref[i]);
    }

    cudaFree(d_indices);
}
