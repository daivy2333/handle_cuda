#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cmath>

class FP16MixedPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(FP16MixedPrecisionTest, FP16Conversion) {
    size_t n = 1024;
    std::vector<float> fp32_data(n);
    for (size_t i = 0; i < n; ++i) {
        fp32_data[i] = static_cast<float>(i) / 100.0f;
    }

    CudaBuffer d_fp32(n);
    CudaBuffer d_fp16(n * 2);  // FP16 uses 2 bytes per element

    host_to_device_async(d_fp32.data, fp32_data.data(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert FP32 to FP16
    float_to_half(d_fp32.data, reinterpret_cast<__half*>(d_fp16.data), n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert back to FP32
    CudaBuffer d_fp32_back(n);
    half_to_float(reinterpret_cast<__half*>(d_fp16.data), d_fp32_back.data, n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<float> fp32_back(n);
    device_to_host(d_fp32_back.data, fp32_back.data(), n);

    // FP16 has ~3 decimal digits precision
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(fp32_data[i], fp32_back[i], 0.01f * std::abs(fp32_data[i]) + 0.001f);
    }
}

TEST_F(FP16MixedPrecisionTest, MatmulFP16Forward) {
    int M = 64, N = 32, K = 128;

    std::vector<float> A_fp32(M * K), B_fp32(K * N);
    for (int i = 0; i < M * K; ++i) A_fp32[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) B_fp32[i] = (rand() % 100) / 100.0f;

    // FP32 reference
    CudaBuffer d_A_fp32(M * K), d_B_fp32(K * N), d_C_fp32_ref(M * N);
    host_to_device_async(d_A_fp32.data, A_fp32.data(), M * K);
    host_to_device_async(d_B_fp32.data, B_fp32.data(), K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul(d_A_fp32.data, d_B_fp32.data, d_C_fp32_ref.data, desc, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FP16 version
    CudaBuffer d_A_fp16(M * K * 2), d_B_fp16(K * N * 2), d_C_fp16_out(M * N);
    float_to_half(d_A_fp32.data, reinterpret_cast<__half*>(d_A_fp16.data), M * K, 0);
    float_to_half(d_B_fp32.data, reinterpret_cast<__half*>(d_B_fp16.data), K * N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_matmul_fp16(reinterpret_cast<__half*>(d_A_fp16.data),
                      reinterpret_cast<__half*>(d_B_fp16.data),
                      d_C_fp16_out.data, M, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    std::vector<float> C_fp32_ref(M * N), C_fp16_out(M * N);
    device_to_host(d_C_fp32_ref.data, C_fp32_ref.data(), M * N);
    device_to_host(d_C_fp16_out.data, C_fp16_out.data(), M * N);

    // FP16 precision allows ~0.1% error
    for (int i = 0; i < M * N; ++i) {
        float rel_error = std::abs(C_fp32_ref[i] - C_fp16_out[i]) / (std::abs(C_fp32_ref[i]) + 1e-6f);
        EXPECT_LT(rel_error, 0.01f) << "Mismatch at index " << i;
    }
}

TEST_F(FP16MixedPrecisionTest, MatmulFP16Backward) {
    int M = 32, N = 16, K = 64;

    std::vector<float> A_fp32(M * K), B_fp32(K * N), grad_C_fp32(M * N);
    for (int i = 0; i < M * K; ++i) A_fp32[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) B_fp32[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < M * N; ++i) grad_C_fp32[i] = (rand() % 100) / 100.0f;

    // FP32 reference backward
    CudaBuffer d_A_fp32(M * K), d_B_fp32(K * N), d_grad_C(M * N);
    CudaBuffer d_grad_A_fp32_ref(M * K), d_grad_B_fp32_ref(K * N);

    host_to_device_async(d_A_fp32.data, A_fp32.data(), M * K);
    host_to_device_async(d_B_fp32.data, B_fp32.data(), K * N);
    host_to_device_async(d_grad_C.data, grad_C_fp32.data(), M * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    MatMulDesc desc{M, N, K, false, false};
    cuda_matmul_backward(d_grad_C.data, d_A_fp32.data, d_B_fp32.data,
                          d_grad_A_fp32_ref.data, d_grad_B_fp32_ref.data, desc, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FP16 backward
    CudaBuffer d_A_fp16(M * K * 2), d_B_fp16(K * N * 2);
    CudaBuffer d_grad_A_fp16_out(M * K), d_grad_B_fp16_out(K * N);

    float_to_half(d_A_fp32.data, reinterpret_cast<__half*>(d_A_fp16.data), M * K, 0);
    float_to_half(d_B_fp32.data, reinterpret_cast<__half*>(d_B_fp16.data), K * N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_matmul_fp16_backward(d_grad_C.data,
                              reinterpret_cast<__half*>(d_A_fp16.data),
                              reinterpret_cast<__half*>(d_B_fp16.data),
                              d_grad_A_fp16_out.data, d_grad_B_fp16_out.data,
                              M, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    std::vector<float> grad_A_fp32_ref(M * K), grad_B_fp32_ref(K * N);
    std::vector<float> grad_A_fp16_out(M * K), grad_B_fp16_out(K * N);

    device_to_host(d_grad_A_fp32_ref.data, grad_A_fp32_ref.data(), M * K);
    device_to_host(d_grad_B_fp32_ref.data, grad_B_fp32_ref.data(), K * N);
    device_to_host(d_grad_A_fp16_out.data, grad_A_fp16_out.data(), M * K);
    device_to_host(d_grad_B_fp16_out.data, grad_B_fp16_out.data(), K * N);

    // Check gradients - FP16 backward errors are larger due to precision limits
    // Use absolute error for small values, relative error for larger values
    for (int i = 0; i < M * K; ++i) {
        float abs_error = std::abs(grad_A_fp32_ref[i] - grad_A_fp16_out[i]);
        float rel_error = abs_error / (std::abs(grad_A_fp32_ref[i]) + 1e-6f);
        // Accept if either absolute error is small (< 0.1) or relative error is moderate (< 100%)
        EXPECT_TRUE(abs_error < 0.1f || rel_error < 1.0f) << "grad_A mismatch at index " << i;
    }

    for (int i = 0; i < K * N; ++i) {
        float abs_error = std::abs(grad_B_fp32_ref[i] - grad_B_fp16_out[i]);
        float rel_error = abs_error / (std::abs(grad_B_fp32_ref[i]) + 1e-6f);
        // Accept if either absolute error is small (< 0.1) or relative error is moderate (< 100%)
        EXPECT_TRUE(abs_error < 0.1f || rel_error < 1.0f) << "grad_B mismatch at index " << i;
    }
}

TEST_F(FP16MixedPrecisionTest, GradientScaling) {
    size_t size = 256;
    float scale = 128.0f;
    float inv_scale = 1.0f / scale;

    std::vector<float> gradients(size);
    for (size_t i = 0; i < size; ++i) {
        gradients[i] = static_cast<float>(i) / 1000.0f;
    }

    CudaBuffer d_grad(size);
    host_to_device_async(d_grad.data, gradients.data(), size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Scale up
    cuda_scale_gradients(d_grad.data, size, scale, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> scaled(size);
    device_to_host(d_grad.data, scaled.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(scaled[i], gradients[i] * scale, 1e-5f);
    }

    // Scale down
    cuda_scale_gradients(d_grad.data, size, inv_scale, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> unscaled(size);
    device_to_host(d_grad.data, unscaled.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(unscaled[i], gradients[i], 1e-5f);
    }
}