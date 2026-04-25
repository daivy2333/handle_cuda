#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>

class SGDTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    std::vector<float> generate_random(size_t size) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = -1.0f + 2.0f * rand() / RAND_MAX;
        }
        return v;
    }
};

TEST_F(SGDTest, BasicUpdate) {
    size_t size = 1024;
    float lr = 0.01f;

    auto param = generate_random(size);
    auto grad = generate_random(size);

    std::vector<float> expected(size);
    for (size_t i = 0; i < size; ++i) {
        expected[i] = param[i] - lr * grad[i];
    }

    CudaBuffer d_param(size), d_grad(size);
    host_to_device_async(d_param.data, param.data(), size);
    host_to_device_async(d_grad.data, grad.data(), size);

    cuda_sgd_update(d_param.data, d_grad.data, size, lr);

    std::vector<float> result(size);
    device_to_host(d_param.data, result.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5f);
    }
}

TEST_F(SGDTest, ZeroGrad) {
    size_t size = 100;
    float lr = 0.1f;

    auto param = generate_random(size);
    std::vector<float> grad(size, 0.0f);

    CudaBuffer d_param(size), d_grad(size);
    host_to_device_async(d_param.data, param.data(), size);
    host_to_device_async(d_grad.data, grad.data(), size);

    cuda_sgd_update(d_param.data, d_grad.data, size, lr);

    std::vector<float> result(size);
    device_to_host(d_param.data, result.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], param[i]);
    }
}

TEST_F(SGDTest, LargeLearningRate) {
    size_t size = 50;
    float lr = 10.0f;

    std::vector<float> param(size, 1.0f);
    std::vector<float> grad(size, 0.1f);

    CudaBuffer d_param(size), d_grad(size);
    host_to_device_async(d_param.data, param.data(), size);
    host_to_device_async(d_grad.data, grad.data(), size);

    cuda_sgd_update(d_param.data, d_grad.data, size, lr);

    std::vector<float> result(size);
    device_to_host(d_param.data, result.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(result[i], 0.0f, 1e-5f);
    }
}