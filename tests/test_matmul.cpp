#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>

class MatMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    float relative_error(float a, float b) {
        if (std::abs(a) < 1e-6 && std::abs(b) < 1e-6) return 0.0f;
        return std::abs(a - b) / (std::abs(a) + std::abs(b) + 1e-6);
    }

    std::vector<float> generate_random(size_t size, float lo = -1.0f, float hi = 1.0f) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
        }
        return v;
    }
};

TEST_F(MatMulTest, SimpleMatMul) {
    size_t M = 128, N = 64, K = 128;

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

    EXPECT_LT(max_rel_err, 1e-5f);
}

TEST_F(MatMulTest, TransposeA) {
    size_t M = 64, N = 128, K = 64;

    auto A = generate_random(K * M);
    auto B = generate_random(K * N);
    std::vector<float> C_ref(M * N, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    CudaBuffer d_A(K * M), d_B(K * N), d_C(M * N);
    host_to_device_async(d_A.data, A.data(), K * M);
    host_to_device_async(d_B.data, B.data(), K * N);

    MatMulDesc desc{M, N, K, true, false};
    cuda_matmul(d_A.data, d_B.data, d_C.data, desc);

    std::vector<float> C_result(M * N);
    device_to_host(d_C.data, C_result.data(), M * N);

    float max_rel_err = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        max_rel_err = std::max(max_rel_err, relative_error(C_ref[i], C_result[i]));
    }

    EXPECT_LT(max_rel_err, 1e-5f);
}

TEST_F(MatMulTest, LargeMatrix) {
    size_t M = 512, N = 512, K = 512;

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

    EXPECT_LT(max_rel_err, 1e-4f);
}
