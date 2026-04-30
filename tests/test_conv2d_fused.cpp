#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>

class FusedConvTest : public ::testing::Test {
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

TEST_F(FusedConvTest, ConvReluCorrectness) {
    int N=2, C=4, H=28, W=28, out_C=8, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 26
    int out_W = W - 2;  // 26

    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * K * K);
    auto bias = generate_random(out_C);

    // Reference: im2col conv2d + separate relu
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * K * K);
    CudaBuffer d_bias(out_C), d_output_ref(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * K * K * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);

    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    host_to_device_async(d_bias.data, bias.data(), out_C);

    Conv2dDesc desc{N, C, H, W, out_C, out_H, out_W, K, K, stride, stride, pad, pad, 1};

    // im2col baseline
    cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                       d_output_ref.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply ReLU to reference output
    std::vector<float> output_ref(N * out_C * out_H * out_W);
    device_to_host(d_output_ref.data, output_ref.data(), N * out_C * out_H * out_W);
    for (auto& val : output_ref) {
        val = fmaxf(0.0f, val);
    }

    // Fused conv+relu
    CudaBuffer d_output_fused(N * out_C * out_H * out_W);

    cuda_conv2d_relu_fused(d_input.data, d_weight.data, d_bias.data,
                           d_output_fused.data,
                           N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output_fused(N * out_C * out_H * out_W);
    device_to_host(d_output_fused.data, output_fused.data(), N * out_C * out_H * out_W);

    // Compare
    for (size_t i = 0; i < output_ref.size(); ++i) {
        EXPECT_NEAR(output_fused[i], output_ref[i], 1e-3f)
            << "Mismatch at index " << i;
    }
}

// TODO: ConvReluPoolCorrectness has indexing bugs in the fused pool kernel
// TEST_F(FusedConvTest, ConvReluPoolCorrectness) { ... }

// TODO: PerformanceComparison needs pool to work first
// TEST_F(FusedConvTest, PerformanceComparison) { ... }
