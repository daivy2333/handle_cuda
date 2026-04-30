#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <cuda_fp16.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

class TensorCoreTest : public ::testing::Test {
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

TEST_F(TensorCoreTest, FP16Conversion) {
    size_t n = 1024;
    auto input = generate_random(n);

    CudaBuffer d_float(n), d_half(n);
    CudaBuffer d_result(n);

    host_to_device_async(d_float.data, input.data(), n);

    // Float -> Half
    float_to_half(d_float.data, (__half*)d_half.data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Half -> Float
    half_to_float((__half*)d_half.data, d_result.data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(n);
    device_to_host(d_result.data, result.data(), n);

    // Check precision loss
    for (size_t i = 0; i < n; ++i) {
        float expected = input[i];
        float actual = result[i];
        // FP16 relative error should be within 1%
        EXPECT_NEAR(actual, expected, std::max(0.01f, std::abs(expected) * 0.01f))
            << "Mismatch at index " << i;
    }
}

TEST_F(TensorCoreTest, MatmulCorrectness) {
    int M = 256, N = 256, K = 256;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);

    CudaBuffer d_A(M * K), d_B(K * N);
    CudaBuffer d_C_fp32(M * N), d_C_tensor(M * N);

    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);

    // FP32 baseline
    cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C_fp32.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FP16 + Tensor Core
    CudaBuffer d_A_half(M * K), d_B_half(K * N);
    float_to_half(d_A.data, (__half*)d_A_half.data, M * K);
    float_to_half(d_B.data, (__half*)d_B_half.data, K * N);

    cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                     d_C_tensor.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_fp32(M * N), C_tensor(M * N);
    device_to_host(d_C_fp32.data, C_fp32.data(), M * N);
    device_to_host(d_C_tensor.data, C_tensor.data(), M * N);

    // Compare results using absolute error for FP16
    // FP16 accumulated errors can be significant for large matmuls
    // We use a combination of relative and absolute tolerance
    for (size_t i = 0; i < M * N; ++i) {
        float expected = C_fp32[i];
        float actual = C_tensor[i];
        float abs_error = std::abs(actual - expected);
        float rel_error = abs_error / (std::abs(expected) + 1e-5f);
        // Accept if either relative error < 30% OR absolute error < 0.01
        EXPECT_TRUE(rel_error < 0.30f || abs_error < 0.01f)
            << "Mismatch at index " << i << ": expected=" << expected << ", actual=" << actual
            << ", rel_error=" << rel_error << ", abs_error=" << abs_error;
    }
}

TEST_F(TensorCoreTest, MatmulPerformance) {
    int M = 1024, N = 1024, K = 1024;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);

    CudaBuffer d_A(M * K), d_B(K * N);
    CudaBuffer d_C(M * N);

    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);

    // Warmup FP32
    for (int i = 0; i < 10; ++i) {
        cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark FP32
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;

    // FP16 conversion + Tensor Core
    CudaBuffer d_A_half(M * K), d_B_half(K * N);
    float_to_half(d_A.data, (__half*)d_A_half.data, M * K);
    float_to_half(d_B.data, (__half*)d_B_half.data, K * N);

    for (int i = 0; i < 10; ++i) {
        cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                         d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                         d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double fp16_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;

    long long flops = 2LL * M * N * K;
    double fp32_gflops = flops / (fp32_ms * 1e6);
    double fp16_gflops = flops / (fp16_ms * 1e6);

    std::cout << "\n========== Tensor Core Matmul Performance ==========\n";
    std::cout << "  FP32: " << fp32_gflops << " GFLOPS (" << fp32_ms << " ms)\n";
    std::cout << "  FP16+TensorCore: " << fp16_gflops << " GFLOPS (" << fp16_ms << " ms)\n";
    std::cout << "  Speedup: " << fp32_ms / fp16_ms << "x\n";
    std::cout << "=====================================================\n";
}

TEST_F(TensorCoreTest, SmallMatmul) {
    // Test small matrices to ensure boundary conditions work
    int M = 64, N = 128, K = 64;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);

    CudaBuffer d_A(M * K), d_B(K * N);
    CudaBuffer d_C_fp32(M * N), d_C_tensor(M * N);

    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);

    cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C_fp32.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaBuffer d_A_half(M * K), d_B_half(K * N);
    float_to_half(d_A.data, (__half*)d_A_half.data, M * K);
    float_to_half(d_B.data, (__half*)d_B_half.data, K * N);

    cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                     d_C_tensor.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_fp32(M * N), C_tensor(M * N);
    device_to_host(d_C_fp32.data, C_fp32.data(), M * N);
    device_to_host(d_C_tensor.data, C_tensor.data(), M * N);

    for (size_t i = 0; i < M * N; ++i) {
        float expected = C_fp32[i];
        float actual = C_tensor[i];
        float abs_error = std::abs(actual - expected);
        float rel_error = abs_error / (std::abs(expected) + 1e-5f);
        EXPECT_TRUE(rel_error < 0.30f || abs_error < 0.01f)
            << "Mismatch at index " << i;
    }
}
