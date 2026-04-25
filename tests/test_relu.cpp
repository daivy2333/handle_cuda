#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>

class ReLUTest : public ::testing::Test {
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

TEST_F(ReLUTest, ForwardBasic) {
    size_t size = 1024;
    auto input = generate_random(size);
    std::vector<float> output_ref(size);

    for (size_t i = 0; i < size; ++i) {
        output_ref[i] = std::fmaxf(0.0f, input[i]);
    }

    CudaBuffer d_input(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_relu(d_input.data, size);

    std::vector<float> output(size);
    device_to_host(d_input.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], output_ref[i]);
    }
}

TEST_F(ReLUTest, BackwardBasic) {
    size_t size = 1024;
    auto input = generate_random(size);
    auto grad_out = generate_random(size);
    std::vector<float> grad_in_ref(size);

    for (size_t i = 0; i < size; ++i) {
        grad_in_ref[i] = input[i] > 0.0f ? grad_out[i] : 0.0f;
    }

    CudaBuffer d_input(size), d_grad_out(size), d_grad_in(size);
    host_to_device_async(d_input.data, input.data(), size);
    host_to_device_async(d_grad_out.data, grad_out.data(), size);

    cuda_relu_backward(d_grad_out.data, d_input.data, d_grad_in.data, size);

    std::vector<float> grad_in(size);
    device_to_host(d_grad_in.data, grad_in.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(grad_in[i], grad_in_ref[i]);
    }
}

TEST_F(ReLUTest, AllPositive) {
    size_t size = 256;
    auto input = generate_random(size);
    for (auto& x : input) x = std::abs(x) + 0.1f;

    CudaBuffer d_input(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_relu(d_input.data, size);

    std::vector<float> output(size);
    device_to_host(d_input.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

TEST_F(ReLUTest, AllNegative) {
    size_t size = 256;
    auto input = generate_random(size);
    for (auto& x : input) x = -std::abs(x) - 0.1f;

    CudaBuffer d_input(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_relu(d_input.data, size);

    std::vector<float> output(size);
    device_to_host(d_input.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }
}
