#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>

class FlattenTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(FlattenTest, ForwardBasic) {
    int batch = 2, C = 3, H = 4, W = 4;
    int flat_size = C * H * W;  // 48

    std::vector<float> input(batch * C * H * W);
    for (int i = 0; i < batch * C * H * W; ++i) {
        input[i] = static_cast<float>(i);
    }

    CudaBuffer d_input(batch * C * H * W), d_output(batch * flat_size);
    host_to_device_async(d_input.data, input.data(), batch * C * H * W);

    cuda_flatten(d_input.data, d_output.data, batch, C, H, W);

    std::vector<float> output(batch * flat_size);
    device_to_host(d_output.data, output.data(), batch * flat_size);

    for (int i = 0; i < batch * C * H * W; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}

TEST_F(FlattenTest, BackwardBasic) {
    int batch = 2, C = 3, H = 4, W = 4;
    int flat_size = C * H * W;

    std::vector<float> grad_flat(batch * flat_size);
    for (int i = 0; i < batch * flat_size; ++i) {
        grad_flat[i] = static_cast<float>(i) * 0.5f;
    }

    CudaBuffer d_grad_flat(batch * flat_size), d_grad_input(batch * C * H * W);
    host_to_device_async(d_grad_flat.data, grad_flat.data(), batch * flat_size);

    cuda_flatten_backward(d_grad_flat.data, d_grad_input.data, batch, C, H, W);

    std::vector<float> grad_input(batch * C * H * W);
    device_to_host(d_grad_input.data, grad_input.data(), batch * C * H * W);

    for (int i = 0; i < batch * C * H * W; ++i) {
        EXPECT_FLOAT_EQ(grad_input[i], grad_flat[i]);
    }
}

TEST_F(FlattenTest, MNISTSize) {
    int batch = 64, C = 1, H = 28, W = 28;
    int flat_size = C * H * W;  // 784

    std::vector<float> input(batch * C * H * W);
    for (int i = 0; i < batch * C * H * W; ++i) {
        input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    CudaBuffer d_input(batch * C * H * W), d_output(batch * flat_size);
    host_to_device_async(d_input.data, input.data(), batch * C * H * W);

    cuda_flatten(d_input.data, d_output.data, batch, C, H, W);

    std::vector<float> output(batch * flat_size);
    device_to_host(d_output.data, output.data(), batch * flat_size);

    for (int i = 0; i < batch * C * H * W; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]);
    }
}