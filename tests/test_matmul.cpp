#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

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

    EXPECT_LT(max_rel_err, 1e-2f);  // Tiled kernel has different FP accumulation order
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

    EXPECT_LT(max_rel_err, 1e-3f);  // Transpose kernel tolerance
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

    EXPECT_LT(max_rel_err, 1e-2f);  // Large matrices accumulate more FP error
}

TEST_F(MatMulTest, BackwardPass) {
    size_t M = 64, N = 32, K = 64;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);
    auto grad_C = generate_random(M * N);

    // Reference backward: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
    std::vector<float> grad_A_ref(M * K, 0.0f);
    std::vector<float> grad_B_ref(K * N, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; ++j) {
                sum += grad_C[i * N + j] * B[k * N + j];
            }
            grad_A_ref[i * K + k] = sum;
        }
    }

    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < M; ++i) {
                sum += A[i * K + k] * grad_C[i * N + j];
            }
            grad_B_ref[k * N + j] = sum;
        }
    }

    CudaBuffer d_A(M * K), d_B(K * N), d_grad_C(M * N);
    CudaBuffer d_grad_A(M * K), d_grad_B(K * N);

    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    host_to_device_async(d_grad_C.data, grad_C.data(), M * N);

    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul_backward(d_grad_C.data, d_A.data, d_B.data,
                         d_grad_A.data, d_grad_B.data, desc);

    std::vector<float> grad_A(M * K), grad_B(K * N);
    device_to_host(d_grad_A.data, grad_A.data(), M * K);
    device_to_host(d_grad_B.data, grad_B.data(), K * N);

    float max_err_A = 0.0f, max_err_B = 0.0f;
    for (size_t i = 0; i < M * K; ++i) {
        max_err_A = std::max(max_err_A, relative_error(grad_A[i], grad_A_ref[i]));
    }
    for (size_t i = 0; i < K * N; ++i) {
        max_err_B = std::max(max_err_B, relative_error(grad_B[i], grad_B_ref[i]));
    }

    EXPECT_LT(max_err_A, 1e-4f);
    EXPECT_LT(max_err_B, 1e-4f);
}

TEST_F(MatMulTest, PerformanceBenchmark) {
    size_t M = 1024, N = 1024, K = 1024;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);

    CudaBuffer d_A(M * K), d_B(K * N), d_C(M * N);
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);

    MatMulDesc desc{M, N, K, false, false};

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_matmul(d_A.data, d_B.data, d_C.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        cuda_matmul(d_A.data, d_B.data, d_C.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double gflops = 2.0 * M * N * K / (elapsed_ms * 1e-3) / 1e9;

    std::cout << "MatMul Performance (M=N=K=1024):\n";
    std::cout << "  Time: " << elapsed_ms << " ms\n";
    std::cout << "  GFLOPS: " << gflops << "\n";

    EXPECT_GT(gflops, 30.0);  // 期望至少 30 GFLOPS
}
