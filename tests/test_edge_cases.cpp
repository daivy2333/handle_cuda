#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>

class EdgeCaseTest : public ::testing::Test {
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

    float relative_error(float a, float b) {
        if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f) return 0.0f;
        return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-6f);
    }

    bool has_nan_or_inf(const std::vector<float>& v) {
        for (float val : v) {
            if (std::isnan(val) || std::isinf(val)) return true;
        }
        return false;
    }
};

// ============================================================================
// Non-square matrix tests
// ============================================================================

TEST_F(EdgeCaseTest, MatMulNonSquare) {
    size_t M = 512, N = 2048, K = 256;
    auto A = generate_random(M * K);
    auto B = generate_random(K * N);
    std::vector<float> C_ref(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }
    CudaBuffer d_A(M * K), d_B(K * N), d_C(M * N);
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul(d_A.data, d_B.data, d_C.data, desc);
    std::vector<float> C_result(M * N);
    device_to_host(d_C.data, C_result.data(), M * N);
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        max_rel_err = std::max(max_rel_err, relative_error(C_ref[i], C_result[i]));
    }
    // Large non-square matrices accumulate more FP error (tiled kernel vs naive CPU)
    EXPECT_LT(max_rel_err, 0.1f) << "Non-square MatMul: M=" << M << ", N=" << N << ", K=" << K;
}

TEST_F(EdgeCaseTest, Conv2dNonSquareInput) {
    int N = 2, C = 4, H = 32, W = 64;
    int out_C = 8, kernel = 3, stride = 1, pad = 1;
    int out_H = (H + 2 * pad - kernel) / stride + 1;
    int out_W = (W + 2 * pad - kernel) / stride + 1;
    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * kernel * kernel);
    auto bias = generate_random(out_C);
    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W, kernel, kernel, stride, stride, pad, pad, 1};
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * kernel * kernel);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * kernel * kernel);
    host_to_device_async(d_bias.data, bias.data(), out_C);
    cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);
    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);
    EXPECT_FALSE(has_nan_or_inf(output));
    EXPECT_EQ(output.size(), static_cast<size_t>(N * out_C * out_H * out_W));
}

// ============================================================================
// batch_size boundary tests
// ============================================================================

TEST_F(EdgeCaseTest, SoftmaxBatchSizeOne) {
    size_t batch_size = 1, num_classes = 100;
    auto input = generate_random(batch_size * num_classes);
    std::vector<float> output_ref(batch_size * num_classes);
    float max_val = -INFINITY;
    for (size_t i = 0; i < num_classes; ++i) {
        max_val = std::fmax(max_val, input[i]);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < num_classes; ++i) {
        sum += std::exp(input[i] - max_val);
    }
    for (size_t i = 0; i < num_classes; ++i) {
        output_ref[i] = std::exp(input[i] - max_val) / sum;
    }
    CudaBuffer d_input(batch_size * num_classes), d_output(batch_size * num_classes);
    host_to_device_async(d_input.data, input.data(), batch_size * num_classes);
    cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);
    std::vector<float> output(batch_size * num_classes);
    device_to_host(d_output.data, output.data(), batch_size * num_classes);
    for (size_t i = 0; i < batch_size * num_classes; ++i) {
        EXPECT_NEAR(output[i], output_ref[i], 1e-5f);
    }
    float prob_sum = 0.0f;
    for (float val : output) prob_sum += val;
    EXPECT_NEAR(prob_sum, 1.0f, 1e-6f);
}

TEST_F(EdgeCaseTest, Conv2dBatchSizeOne) {
    int N = 1, C = 16, H = 28, W = 28;
    int out_C = 32, kernel = 5, stride = 1, pad = 2;
    int out_H = (H + 2 * pad - kernel) / stride + 1;
    int out_W = (W + 2 * pad - kernel) / stride + 1;
    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * kernel * kernel);
    auto bias = generate_random(out_C);
    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W, kernel, kernel, stride, stride, pad, pad, 1};
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * kernel * kernel);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * kernel * kernel);
    host_to_device_async(d_bias.data, bias.data(), out_C);
    cuda_conv2d(d_input.data, d_weight.data, d_bias.data, d_output.data, desc);
    std::vector<float> output(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output.data(), N * out_C * out_H * out_W);
    EXPECT_FALSE(has_nan_or_inf(output));
}

// ============================================================================
// Memory pressure tests
// ============================================================================

TEST_F(EdgeCaseTest, MatMulLargeMatrix) {
    size_t M = 4096, N = 4096, K = 4096;
    auto A = generate_random(M * K);
    auto B = generate_random(K * N);
    CudaBuffer d_A(M * K), d_B(K * N), d_C(M * N);
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul(d_A.data, d_B.data, d_C.data, desc);
    std::vector<float> C_result(M * N);
    device_to_host(d_C.data, C_result.data(), M * N);
    EXPECT_FALSE(has_nan_or_inf(C_result));
    EXPECT_EQ(C_result.size(), M * N);
}

TEST_F(EdgeCaseTest, MatMulVeryLarge8K) {
    size_t M = 8192, N = 8192, K = 8192;
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_mem = (M * K + K * N + M * N) * sizeof(float);
    if (free_mem < required_mem) {
        GTEST_SKIP() << "Insufficient GPU memory for 8K matrices (need "
                     << required_mem / 1024 / 1024 << " MB, have "
                     << free_mem / 1024 / 1024 << " MB free)";
    }
    auto A = generate_random(M * K);
    auto B = generate_random(K * N);
    CudaBuffer d_A(M * K), d_B(K * N), d_C(M * N);
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul(d_A.data, d_B.data, d_C.data, desc);
    std::vector<float> C_result(M * N);
    device_to_host(d_C.data, C_result.data(), M * N);
    EXPECT_FALSE(has_nan_or_inf(C_result));
}

// ============================================================================
// NaN/Inf tolerance tests
// ============================================================================

TEST_F(EdgeCaseTest, SoftmaxWithNaNInput) {
    size_t batch_size = 4, num_classes = 10;
    auto input = generate_random(batch_size * num_classes);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CudaBuffer d_input(batch_size * num_classes), d_output(batch_size * num_classes);
    host_to_device_async(d_input.data, input.data(), batch_size * num_classes);
    cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);
    std::vector<float> output(batch_size * num_classes);
    device_to_host(d_output.data, output.data(), batch_size * num_classes);
    bool first_batch_has_nan = false;
    for (size_t i = 0; i < num_classes; ++i) {
        if (std::isnan(output[i])) first_batch_has_nan = true;
    }
    EXPECT_TRUE(first_batch_has_nan);
    for (size_t b = 1; b < batch_size; ++b) {
        for (size_t i = 0; i < num_classes; ++i) {
            EXPECT_FALSE(std::isnan(output[b * num_classes + i]));
        }
    }
}

TEST_F(EdgeCaseTest, SoftmaxWithInfInput) {
    size_t batch_size = 2, num_classes = 5;
    auto input = generate_random(batch_size * num_classes);
    input[0] = std::numeric_limits<float>::infinity();
    CudaBuffer d_input(batch_size * num_classes), d_output(batch_size * num_classes);
    host_to_device_async(d_input.data, input.data(), batch_size * num_classes);
    cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);
    std::vector<float> output(batch_size * num_classes);
    device_to_host(d_output.data, output.data(), batch_size * num_classes);
    EXPECT_NEAR(output[0], 1.0f, 1e-5f);
    for (size_t i = 1; i < num_classes; ++i) {
        EXPECT_NEAR(output[i], 0.0f, 1e-5f);
    }
    float sum = 0.0f;
    for (size_t i = num_classes; i < 2 * num_classes; ++i) {
        sum += output[i];
        EXPECT_FALSE(std::isnan(output[i]));
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST_F(EdgeCaseTest, ReLUWithNaNInput) {
    size_t size = 100;
    auto input = generate_random(size);
    input[10] = std::numeric_limits<float>::quiet_NaN();
    input[50] = -std::numeric_limits<float>::infinity();
    CudaBuffer d_data(size);
    host_to_device_async(d_data.data, input.data(), size);
    cuda_relu(d_data.data, size);
    std::vector<float> output(size);
    device_to_host(d_data.data, output.data(), size);
    EXPECT_TRUE(std::isnan(output[10]));
    EXPECT_NEAR(output[50], 0.0f, 1e-6f);
    for (size_t i = 0; i < size; ++i) {
        if (i != 10 && i != 50) {
            float expected = input[i] > 0 ? input[i] : 0.0f;
            EXPECT_NEAR(output[i], expected, 1e-5f);
        }
    }
}