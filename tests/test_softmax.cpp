#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

class SoftmaxTest : public ::testing::Test {
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

TEST_F(SoftmaxTest, Basic) {
    size_t batch_size = 32, num_classes = 10;
    size_t size = batch_size * num_classes;

    auto input = generate_random(size);
    std::vector<float> output_ref(size);

    for (size_t b = 0; b < batch_size; ++b) {
        const float* in_row = input.data() + b * num_classes;
        float* out_row = output_ref.data() + b * num_classes;

        float max_val = -INFINITY;
        for (size_t i = 0; i < num_classes; ++i) {
            max_val = std::fmax(max_val, in_row[i]);
        }

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            sum += std::exp(in_row[i] - max_val);
        }

        for (size_t i = 0; i < num_classes; ++i) {
            out_row[i] = std::exp(in_row[i] - max_val) / sum;
        }
    }

    CudaBuffer d_input(size), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(output[i], output_ref[i], 1e-5f);
    }
}

TEST_F(SoftmaxTest, SumToOne) {
    size_t batch_size = 16, num_classes = 5;
    size_t size = batch_size * num_classes;

    auto input = generate_random(size);

    CudaBuffer d_input(size), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    for (size_t b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            sum += output[b * num_classes + i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

TEST_F(SoftmaxTest, NonNegative) {
    size_t batch_size = 8, num_classes = 10;
    size_t size = batch_size * num_classes;

    auto input = generate_random(size);

    CudaBuffer d_input(size), d_output(size);
    host_to_device_async(d_input.data, input.data(), size);

    cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);

    std::vector<float> output(size);
    device_to_host(d_output.data, output.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_GE(output[i], 0.0f);
    }
}

TEST_F(SoftmaxTest, PerformanceBenchmark) {
    size_t batch_size = 256;
    size_t num_classes = 1000;

    auto input = generate_random(batch_size * num_classes);

    CudaBuffer d_input(batch_size * num_classes), d_output(batch_size * num_classes);
    host_to_device_async(d_input.data, input.data(), batch_size * num_classes);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 1000;
    for (int i = 0; i < iterations; ++i) {
        cuda_softmax(d_input.data, d_output.data, batch_size, num_classes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double bandwidth = batch_size * num_classes * sizeof(float) * 2 / (elapsed_ms * 1e-3) / 1e9;

    std::cout << "Softmax Performance (batch=" << batch_size << ", classes=" << num_classes << "):\n";
    std::cout << "  Time: " << elapsed_ms << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " GB/s\n";
}
