#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>

class TanhTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        srand(42);  // 固定随机种子，使测试可复现
    }

    float relative_error(float a, float b) {
        if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f) return 0.0f;
        return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-6f);
    }

    std::vector<float> generate_random(size_t size, float lo = -5.0f, float hi = 5.0f) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
        }
        return v;
    }

    std::vector<float> tanh_ref(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::tanh(input[i]);
        }
        return output;
    }

    std::vector<float> tanh_backward_ref(const std::vector<float>& grad_out,
                                         const std::vector<float>& forward_output) {
        std::vector<float> grad_in(grad_out.size());
        for (size_t i = 0; i < grad_out.size(); ++i) {
            grad_in[i] = grad_out[i] * (1.0f - forward_output[i] * forward_output[i]);
        }
        return grad_in;
    }
};

TEST_F(TanhTest, BasicForward) {
    size_t size = 1024;
    auto input = generate_random(size);
    auto expected = tanh_ref(input);

    CudaBuffer d_input(size), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_tanh(d_input.data, d_output.data, size);

    std::vector<float> result(size);
    device_to_host(d_output.data, result.data(), size);

    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_err = std::max(max_err, relative_error(result[i], expected[i]));
    }
    EXPECT_LT(max_err, 1e-5f);
}

TEST_F(TanhTest, BasicBackward) {
    size_t size = 1024;
    auto forward_input = generate_random(size);
    auto forward_output = tanh_ref(forward_input);
    auto grad_out = generate_random(size);
    auto expected = tanh_backward_ref(grad_out, forward_output);

    CudaBuffer d_grad_out(size), d_forward_output(size), d_grad_in(size);
    host_to_device_async(d_grad_out.data, grad_out.data(), size);
    host_to_device_async(d_forward_output.data, forward_output.data(), size);

    cuda_tanh_backward(d_grad_out.data, d_forward_output.data, d_grad_in.data, size);

    std::vector<float> result(size);
    device_to_host(d_grad_in.data, result.data(), size);

    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_err = std::max(max_err, relative_error(result[i], expected[i]));
    }
    EXPECT_LT(max_err, 1e-4f);
}

TEST_F(TanhTest, LargeSize) {
    size_t size = 1024 * 1024;
    auto input = generate_random(size);
    auto expected = tanh_ref(input);

    CudaBuffer d_input(size), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_tanh(d_input.data, d_output.data, size);

    std::vector<float> result(size);
    device_to_host(d_output.data, result.data(), size);

    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_err = std::max(max_err, relative_error(result[i], expected[i]));
    }
    EXPECT_LT(max_err, 1e-4f);
}

TEST_F(TanhTest, ExtremeValues) {
    std::vector<float> input = {-10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f};
    auto expected = tanh_ref(input);

    CudaBuffer d_input(input.size()), d_output(input.size());
    host_to_device_async(d_input.data, input.data(), input.size());

    cuda_tanh(d_input.data, d_output.data, input.size());

    std::vector<float> result(input.size());
    device_to_host(d_output.data, result.data(), input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_LT(std::abs(result[i] - expected[i]), 1e-6f);
    }
}