#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>

class DropoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        srand(42);
    }

    std::vector<float> generate_random(size_t size, float lo = -1.0f, float hi = 1.0f) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
        }
        return v;
    }
};

TEST_F(DropoutTest, TrainingModeDropout) {
    size_t size = 10000;
    float dropout_prob = 0.5f;

    auto input = generate_random(size);

    CudaBuffer d_input(size), d_output(size), d_mask(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_dropout(d_input.data, d_output.data, d_mask.data, size, dropout_prob, true);

    std::vector<float> output(size), mask(size);
    device_to_host(d_output.data, output.data(), size);
    device_to_host(d_mask.data, mask.data(), size);

    int zeros = 0;
    for (size_t i = 0; i < size; ++i) {
        if (mask[i] == 0.0f) {
            zeros++;
            EXPECT_FLOAT_EQ(output[i], 0.0f);
        } else {
            float scale = 1.0f / (1.0f - dropout_prob);
            EXPECT_NEAR(output[i], input[i] * scale, 1e-5f);
        }
    }

    float drop_ratio = zeros / static_cast<float>(size);
    EXPECT_NEAR(drop_ratio, dropout_prob, 0.05f);
}

TEST_F(DropoutTest, InferenceModeNoDropout) {
    size_t size = 1000;
    float dropout_prob = 0.5f;

    auto input = generate_random(size);

    CudaBuffer d_input(size), d_output(size), d_mask(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_dropout(d_input.data, d_output.data, d_mask.data, size, dropout_prob, false);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

TEST_F(DropoutTest, BackwardPass) {
    size_t size = 1000;
    float dropout_prob = 0.3f;

    auto input = generate_random(size);
    auto grad_out = generate_random(size);

    CudaBuffer d_input(size), d_output(size), d_mask(size), d_grad_out(size), d_grad_in(size);

    host_to_device_async(d_input.data, input.data(), size);
    cuda_dropout(d_input.data, d_output.data, d_mask.data, size, dropout_prob, true);

    host_to_device_async(d_grad_out.data, grad_out.data(), size);
    cuda_dropout_backward(d_grad_out.data, d_mask.data, d_grad_in.data, size);

    std::vector<float> mask(size), grad_in(size);
    device_to_host(d_mask.data, mask.data(), size);
    device_to_host(d_grad_in.data, grad_in.data(), size);

    for (size_t i = 0; i < size; ++i) {
        if (mask[i] == 0.0f) {
            EXPECT_FLOAT_EQ(grad_in[i], 0.0f);
        } else {
            EXPECT_NEAR(grad_in[i], grad_out[i] * mask[i], 1e-5f);
        }
    }
}

TEST_F(DropoutTest, ZeroDropoutProb) {
    size_t size = 1000;
    float dropout_prob = 0.0f;

    auto input = generate_random(size);
    CudaBuffer d_input(size), d_output(size), d_mask(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_dropout(d_input.data, d_output.data, d_mask.data, size, dropout_prob, true);

    std::vector<float> output(size), mask(size);
    device_to_host(d_output.data, output.data(), size);
    device_to_host(d_mask.data, mask.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
        EXPECT_FLOAT_EQ(mask[i], 1.0f);
    }
}

TEST_F(DropoutTest, FullDropoutProb) {
    size_t size = 1000;
    float dropout_prob = 1.0f;

    auto input = generate_random(size);
    CudaBuffer d_input(size), d_output(size), d_mask(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_dropout(d_input.data, d_output.data, d_mask.data, size, dropout_prob, true);

    std::vector<float> output(size), mask(size);
    device_to_host(d_output.data, output.data(), size);
    device_to_host(d_mask.data, mask.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
        EXPECT_FLOAT_EQ(mask[i], 0.0f);
    }
}