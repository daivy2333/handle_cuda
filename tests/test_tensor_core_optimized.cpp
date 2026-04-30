#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>
#include <chrono>

class TensorCoreOptimizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }

    std::vector<float> generate_random(size_t n) {
        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = (rand() % 1000) / 1000.0f - 0.5f;
        }
        return data;
    }
};

TEST_F(TensorCoreOptimizedTest, OptimizedKernelCorrectness) {
    int M = 256, N = 256, K = 256;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);

    CudaBuffer d_A_fp32(M * K), d_B_fp32(K * N), d_C_fp32(M * N);
    CudaBuffer d_A_fp16(M * K * 2), d_B_fp16(K * N * 2), d_C_fp16(M * N);

    host_to_device_async(d_A_fp32.data, A.data(), M * K);
    host_to_device_async(d_B_fp32.data, B.data(), K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FP32 reference
    cuda_matmul_fp32_baseline(d_A_fp32.data, d_B_fp32.data, d_C_fp32.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FP16 conversion
    float_to_half(d_A_fp32.data, reinterpret_cast<__half*>(d_A_fp16.data), M * K);
    float_to_half(d_B_fp32.data, reinterpret_cast<__half*>(d_B_fp16.data), K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Tensor Core optimized kernel
    cuda_matmul_fp16_tensor_core(reinterpret_cast<__half*>(d_A_fp16.data),
                                  reinterpret_cast<__half*>(d_B_fp16.data),
                                  d_C_fp16.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    std::vector<float> C_fp32(M * N), C_fp16(M * N);
    device_to_host(d_C_fp32.data, C_fp32.data(), M * N);
    device_to_host(d_C_fp16.data, C_fp16.data(), M * N);

    for (int i = 0; i < M * N; ++i) {
        float rel_error = std::abs(C_fp32[i] - C_fp16[i]) / (std::abs(C_fp32[i]) + 1e-5f);
        EXPECT_LT(rel_error, 0.15f) << "Mismatch at index " << i;
    }
}

TEST_F(TensorCoreOptimizedTest, OptimizedKernelPerformance) {
    int M = 2048, N = 2048, K = 2048;

    auto A = generate_random(M * K);
    auto B = generate_random(K * N);

    CudaBuffer d_A_fp32(M * K), d_B_fp32(K * N), d_C(M * N);
    CudaBuffer d_A_fp16(M * K * 2), d_B_fp16(K * N * 2);

    host_to_device_async(d_A_fp32.data, A.data(), M * K);
    host_to_device_async(d_B_fp32.data, B.data(), K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float_to_half(d_A_fp32.data, reinterpret_cast<__half*>(d_A_fp16.data), M * K);
    float_to_half(d_B_fp32.data, reinterpret_cast<__half*>(d_B_fp16.data), K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    for (int i = 0; i < 5; ++i) {
        cuda_matmul_fp32_baseline(d_A_fp32.data, d_B_fp32.data, d_C.data, M, N, K);
        cuda_matmul_fp16_tensor_core(reinterpret_cast<__half*>(d_A_fp16.data),
                                      reinterpret_cast<__half*>(d_B_fp16.data),
                                      d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark FP32
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        cuda_matmul_fp32_baseline(d_A_fp32.data, d_B_fp32.data, d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(end - start).count() / 50.0;

    // Benchmark Tensor Core Optimized
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        cuda_matmul_fp16_tensor_core(reinterpret_cast<__half*>(d_A_fp16.data),
                                      reinterpret_cast<__half*>(d_B_fp16.data),
                                      d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double tc_ms = std::chrono::duration<double, std::milli>(end - start).count() / 50.0;

    long long flops = 2LL * M * N * K;
    double fp32_gflops = flops / (fp32_ms * 1e6);
    double tc_gflops = flops / (tc_ms * 1e6);

    std::cout << "\n========== Optimized Tensor Core Performance ==========\n";
    std::cout << "  Matrix size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "  FP32 baseline: " << fp32_gflops << " GFLOPS (" << fp32_ms << " ms)\n";
    std::cout << "  Tensor Core optimized: " << tc_gflops << " GFLOPS (" << tc_ms << " ms)\n";
    std::cout << "  Speedup: " << fp32_ms / tc_ms << "x\n";
    std::cout << "========================================================\n";
}