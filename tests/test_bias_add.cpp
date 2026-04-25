#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>

class BiasAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    std::vector<float> generate_random(size_t size, float lo = -1.0f, float hi = 1.0f) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
        }
        return v;
    }
};

TEST_F(BiasAddTest, Basic) {
    size_t rows = 128, cols = 64;
    size_t size = rows * cols;

    auto input = generate_random(size);
    auto bias = generate_random(cols);
    std::vector<float> output_ref(size);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output_ref[i * cols + j] = input[i * cols + j] + bias[j];
        }
    }

    CudaBuffer d_input(size), d_bias(cols), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);
    host_to_device_async(d_bias.data, bias.data(), cols);

    cuda_bias_add(d_input.data, d_bias.data, d_output.data, rows, cols);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], output_ref[i]);
    }
}

TEST_F(BiasAddTest, SingleRow) {
    size_t rows = 1, cols = 128;
    size_t size = rows * cols;

    auto input = generate_random(size);
    auto bias = generate_random(cols);
    std::vector<float> output_ref(size);

    for (size_t j = 0; j < cols; ++j) {
        output_ref[j] = input[j] + bias[j];
    }

    CudaBuffer d_input(size), d_bias(cols), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);
    host_to_device_async(d_bias.data, bias.data(), cols);

    cuda_bias_add(d_input.data, d_bias.data, d_output.data, rows, cols);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(output[i], output_ref[i]);
    }
}

TEST_F(BiasAddTest, LargeMatrix) {
    size_t rows = 1024, cols = 512;
    size_t size = rows * cols;

    auto input = generate_random(size);
    auto bias = generate_random(cols);
    std::vector<float> output_ref(size);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output_ref[i * cols + j] = input[i * cols + j] + bias[j];
        }
    }

    CudaBuffer d_input(size), d_bias(cols), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);
    host_to_device_async(d_bias.data, bias.data(), cols);

    cuda_bias_add(d_input.data, d_bias.data, d_output.data, rows, cols);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    float max_err = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_err = std::max(max_err, std::abs(output[i] - output_ref[i]));
    }

    EXPECT_LT(max_err, 1e-6f);
}

TEST_F(BiasAddTest, BackwardPass) {
    size_t rows = 128, cols = 64;

    auto grad_out = generate_random(rows * cols);

    CudaBuffer d_grad_out(rows * cols), d_grad_input(rows * cols), d_grad_bias(cols);
    host_to_device_async(d_grad_out.data, grad_out.data(), rows * cols);

    cuda_bias_add_backward(d_grad_out.data, d_grad_input.data, d_grad_bias.data, rows, cols);

    std::vector<float> grad_input(rows * cols), grad_bias(cols);
    device_to_host(d_grad_input.data, grad_input.data(), rows * cols);
    device_to_host(d_grad_bias.data, grad_bias.data(), cols);

    // grad_input = grad_out (identity for element-wise addition)
    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(grad_input[i], grad_out[i], 1e-5f);
    }

    // grad_bias = sum over rows
    std::vector<float> expected_grad_bias(cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected_grad_bias[c] += grad_out[r * cols + c];
        }
    }
    for (size_t c = 0; c < cols; ++c) {
        EXPECT_NEAR(grad_bias[c], expected_grad_bias[c], 1e-4f);
    }
}

TEST_F(BiasAddTest, BackwardSingleRow) {
    size_t rows = 1, cols = 64;

    auto grad_out = generate_random(cols);

    CudaBuffer d_grad_out(cols), d_grad_input(cols), d_grad_bias(cols);
    host_to_device_async(d_grad_out.data, grad_out.data(), cols);

    cuda_bias_add_backward(d_grad_out.data, d_grad_input.data, d_grad_bias.data, rows, cols);

    std::vector<float> grad_input(cols), grad_bias(cols);
    device_to_host(d_grad_input.data, grad_input.data(), cols);
    device_to_host(d_grad_bias.data, grad_bias.data(), cols);

    for (size_t i = 0; i < cols; ++i) {
        EXPECT_NEAR(grad_input[i], grad_out[i], 1e-5f);
        EXPECT_NEAR(grad_bias[i], grad_out[i], 1e-5f);
    }
}

TEST_F(BiasAddTest, BackwardLargeMatrix) {
    size_t rows = 1024, cols = 512;

    auto grad_out = generate_random(rows * cols);

    CudaBuffer d_grad_out(rows * cols), d_grad_input(rows * cols), d_grad_bias(cols);
    host_to_device_async(d_grad_out.data, grad_out.data(), rows * cols);

    cuda_bias_add_backward(d_grad_out.data, d_grad_input.data, d_grad_bias.data, rows, cols);

    std::vector<float> grad_input(rows * cols), grad_bias(cols);
    device_to_host(d_grad_input.data, grad_input.data(), rows * cols);
    device_to_host(d_grad_bias.data, grad_bias.data(), cols);

    // Verify grad_input
    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(grad_input[i], grad_out[i], 1e-4f);
    }

    // Verify grad_bias (sum over rows)
    std::vector<float> expected_grad_bias(cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            expected_grad_bias[c] += grad_out[r * cols + c];
        }
    }
    for (size_t c = 0; c < cols; ++c) {
        EXPECT_NEAR(grad_bias[c], expected_grad_bias[c], 1e-3f);
    }
}
