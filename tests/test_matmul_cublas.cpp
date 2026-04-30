#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>

class MatmulCublasTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(MatmulCublasTest, Correctness) {
    int M=512, N=128, K=256;
    std::vector<float> A(M * K), B(K * N), C_cublas(M * N), C_ref(M * N);

    // Initialize random data
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() % 100) / 100.0f;

    CudaBuffer d_A(M * K), d_B(K * N), d_C_cublas(M * N), d_C_ref(M * N);
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    cudaDeviceSynchronize();

    // Call cuBLAS version
    cuda_matmul_cublas(d_A.data, d_B.data, d_C_cublas.data, M, N, K, 0);
    cudaDeviceSynchronize();

    // Call custom implementation for comparison
    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul(d_A.data, d_B.data, d_C_ref.data, desc, 0);
    cudaDeviceSynchronize();

    // Compare results
    std::vector<float> h_c_cublas(M * N), h_c_ref(M * N);
    device_to_host(d_C_cublas.data, h_c_cublas.data(), M * N);
    device_to_host(d_C_ref.data, h_c_ref.data(), M * N);

    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(h_c_cublas[i], h_c_ref[i], 1e-3f);
    }
}

TEST_F(MatmulCublasTest, Performance) {
    int M=2048, N=2048, K=2048;
    std::vector<float> A(M * K), B(K * N), C(M * N);

    for (int i = 0; i < M * K; i++) A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() % 100) / 100.0f;

    CudaBuffer d_A(M * K), d_B(K * N), d_C(M * N);
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cuda_matmul_cublas(d_A.data, d_B.data, d_C.data, M, N, K, 0);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float time_sec = elapsed_ms * 1e-3f;  // Convert to seconds
    float total_flops = 2.0f * M * N * K * 10;  // Total operations
    float gflops = total_flops / time_sec / 1e9f;  // GFLOPS
    printf("cuBLAS MatMul: %.1f GFLOPS (%.2f ms)\n", gflops, elapsed_ms / 10);

    // Target: > 1500 GFLOPS
    EXPECT_GT(gflops, 1500.0f);
}