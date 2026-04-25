#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>

class CrossEntropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
        srand(42);
    }

    float relative_error(float a, float b) {
        if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f) return 0.0f;
        return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-6f);
    }

    std::vector<float> generate_logits(size_t batch, size_t classes) {
        std::vector<float> v(batch * classes);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = -2.0f + 4.0f * rand() / RAND_MAX;
        }
        return v;
    }

    std::vector<int> generate_targets(size_t batch, size_t classes) {
        std::vector<int> v(batch);
        for (size_t i = 0; i < batch; ++i) {
            v[i] = rand() % classes;
        }
        return v;
    }

    float cross_entropy_ref(const std::vector<float>& logits,
                            const std::vector<int>& targets,
                            size_t batch, size_t classes) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < batch; ++i) {
            float max_val = logits[i * classes];
            for (size_t j = 1; j < classes; ++j) {
                max_val = std::max(max_val, logits[i * classes + j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < classes; ++j) {
                sum += std::exp(logits[i * classes + j] - max_val);
            }
            float log_prob = logits[i * classes + targets[i]] - max_val - std::log(sum);
            total_loss -= log_prob;
        }
        return total_loss / batch;
    }

    std::vector<float> cross_entropy_backward_ref(const std::vector<float>& logits,
                                                   const std::vector<int>& targets,
                                                   size_t batch, size_t classes) {
        std::vector<float> grad(batch * classes);
        for (size_t i = 0; i < batch; ++i) {
            float max_val = logits[i * classes];
            for (size_t j = 1; j < classes; ++j) {
                max_val = std::max(max_val, logits[i * classes + j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < classes; ++j) {
                sum += std::exp(logits[i * classes + j] - max_val);
            }
            for (size_t j = 0; j < classes; ++j) {
                float softmax_val = std::exp(logits[i * classes + j] - max_val) / sum;
                grad[i * classes + j] = (softmax_val - (j == targets[i] ? 1.0f : 0.0f)) / batch;
            }
        }
        return grad;
    }
};

TEST_F(CrossEntropyTest, BasicForward) {
    size_t batch = 64, classes = 10;
    auto logits = generate_logits(batch, classes);
    auto targets = generate_targets(batch, classes);
    float expected_loss = cross_entropy_ref(logits, targets, batch, classes);

    CudaBuffer d_logits(batch * classes), d_grad(batch * classes);
    CudaBuffer d_loss(1);
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, batch * sizeof(int)));

    host_to_device_async(d_logits.data, logits.data(), batch * classes);
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

    cuda_cross_entropy_loss(d_logits.data, d_targets, d_loss.data, d_grad.data, batch, classes);

    float loss;
    device_to_host(d_loss.data, &loss, 1);

    EXPECT_LT(std::abs(loss - expected_loss), 1e-4f);
    CUDA_CHECK(cudaFree(d_targets));
}

TEST_F(CrossEntropyTest, BackwardPass) {
    size_t batch = 32, classes = 10;
    auto logits = generate_logits(batch, classes);
    auto targets = generate_targets(batch, classes);
    auto expected_grad = cross_entropy_backward_ref(logits, targets, batch, classes);

    CudaBuffer d_logits(batch * classes), d_grad(batch * classes);
    CudaBuffer d_loss(1);
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, batch * sizeof(int)));

    host_to_device_async(d_logits.data, logits.data(), batch * classes);
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

    cuda_cross_entropy_loss(d_logits.data, d_targets, d_loss.data, d_grad.data, batch, classes);

    std::vector<float> grad(batch * classes);
    device_to_host(d_grad.data, grad.data(), batch * classes);

    float max_err = 0.0f;
    for (size_t i = 0; i < batch * classes; ++i) {
        max_err = std::max(max_err, relative_error(grad[i], expected_grad[i]));
    }
    EXPECT_LT(max_err, 1e-4f);
    CUDA_CHECK(cudaFree(d_targets));
}

TEST_F(CrossEntropyTest, UniformLogits) {
    size_t batch = 1, classes = 10;
    std::vector<float> logits(10, 0.0f);
    std::vector<int> targets = {5};

    float expected_loss = cross_entropy_ref(logits, targets, batch, classes);

    CudaBuffer d_logits(classes), d_grad(classes);
    CudaBuffer d_loss(1);
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, sizeof(int)));

    host_to_device_async(d_logits.data, logits.data(), classes);
    CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), sizeof(int), cudaMemcpyHostToDevice));

    cuda_cross_entropy_loss(d_logits.data, d_targets, d_loss.data, d_grad.data, batch, classes);

    float loss;
    device_to_host(d_loss.data, &loss, 1);

    EXPECT_NEAR(loss, std::log(10.0f), 1e-4f);
    CUDA_CHECK(cudaFree(d_targets));
}