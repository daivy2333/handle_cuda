#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>

class BenchmarkTimer {
public:
    void start() {
        cudaDeviceSynchronize();
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stop(int iterations = 1) {
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count() / iterations;
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

std::vector<float> generate_random(size_t size) {
    std::vector<float> v(size);
    for (size_t i = 0; i < size; ++i) {
        v[i] = -1.0f + 2.0f * rand() / RAND_MAX;
    }
    return v;
}

void benchmark_matmul() {
    std::cout << "\n=== MatMul Benchmark ===\n";

    for (int size : {512, 1024, 2048}) {
        CudaBuffer A(size * size), B(size * size), C(size * size);
        A.clear(); B.clear();

        MatMulDesc desc{static_cast<size_t>(size), static_cast<size_t>(size), static_cast<size_t>(size), false, false};

        // Warmup
        for (int i = 0; i < 10; ++i) cuda_matmul(A.data, B.data, C.data, desc);

        BenchmarkTimer timer;
        timer.start();
        for (int i = 0; i < 100; ++i) cuda_matmul(A.data, B.data, C.data, desc);
        double ms = timer.stop(100);
        double gflops = 2.0 * size * size * size / (ms * 1e-3) / 1e9;

        std::cout << "  Size " << size << "x" << size << ": " << ms << " ms, " << gflops << " GFLOPS\n";
    }
}

void benchmark_softmax() {
    std::cout << "\n=== Softmax Benchmark ===\n";

    for (int classes : {100, 1000, 10000}) {
        int batch = 256;
        CudaBuffer input(batch * classes), output(batch * classes);
        input.clear();

        // Warmup
        for (int i = 0; i < 10; ++i) cuda_softmax(input.data, output.data, batch, classes);

        BenchmarkTimer timer;
        timer.start();
        for (int i = 0; i < 1000; ++i) cuda_softmax(input.data, output.data, batch, classes);
        double ms = timer.stop(1000);
        double bandwidth = batch * classes * sizeof(float) * 2 / (ms * 1e-3) / 1e9;

        std::cout << "  Batch=256, Classes=" << classes << ": " << ms << " ms, " << bandwidth << " GB/s\n";
    }
}

void benchmark_relu() {
    std::cout << "\n=== ReLU Benchmark ===\n";

    for (int size_mb : {1, 10, 100}) {
        size_t size = size_mb * 1024 * 1024;
        CudaBuffer data(size);
        data.clear();

        // Warmup
        for (int i = 0; i < 10; ++i) cuda_relu(data.data, size);

        BenchmarkTimer timer;
        timer.start();
        for (int i = 0; i < 1000; ++i) cuda_relu(data.data, size);
        double ms = timer.stop(1000);
        double bandwidth = size * sizeof(float) * 2 / (ms * 1e-3) / 1e9;

        std::cout << "  Size " << size_mb << "M: " << ms << " ms, " << bandwidth << " GB/s\n";
    }
}

void benchmark_conv2d() {
    std::cout << "\n=== Conv2d Benchmark ===\n";

    // Typical CNN layer sizes
    struct TestCase { int N, C, H, W, out_C, K; };
    TestCase cases[] = {
        {32, 64, 32, 32, 64, 3},   // ResNet block
        {16, 128, 16, 16, 128, 3}, // ResNet block
        {1, 3, 224, 224, 64, 7},   // First conv
    };

    for (auto& tc : cases) {
        int out_H = tc.H - tc.K + 1;
        int out_W = tc.W - tc.K + 1;

        CudaBuffer input(tc.N * tc.C * tc.H * tc.W);
        CudaBuffer weight(tc.out_C * tc.C * tc.K * tc.K);
        CudaBuffer output(tc.N * tc.out_C * out_H * out_W);
        CudaBuffer col_buffer(tc.C * tc.K * tc.K * tc.N * out_H * out_W);
        CudaBuffer gemm_buffer(tc.out_C * tc.N * out_H * out_W);

        input.clear(); weight.clear();

        Conv2dDesc desc{tc.N, tc.C, tc.H, tc.W, tc.out_C, out_H, out_W,
                        tc.K, tc.K, 1, 1, 0, 0, 1};

        // Warmup
        for (int i = 0; i < 10; ++i) cuda_conv2d_im2col(input.data, weight.data, nullptr, output.data, col_buffer.data, gemm_buffer.data, desc);

        BenchmarkTimer timer;
        timer.start();
        for (int i = 0; i < 100; ++i) cuda_conv2d_im2col(input.data, weight.data, nullptr, output.data, col_buffer.data, gemm_buffer.data, desc);
        double ms = timer.stop(100);

        // GFLOPS = 2 * N * out_C * C * K^2 * out_H * out_W
        long long ops = 2LL * tc.N * tc.out_C * tc.C * tc.K * tc.K * out_H * out_W;
        double gflops = ops / (ms * 1e-3) / 1e9;

        std::cout << "  N=" << tc.N << " C=" << tc.C << " H=" << tc.H << " W=" << tc.W
                  << " out_C=" << tc.out_C << " K=" << tc.K << ": " << ms << " ms, " << gflops << " GFLOPS\n";
    }
}

int main() {
    std::cout << "CUDA Deep Learning Operators Benchmark\n";
    std::cout << "========================================\n";

    benchmark_matmul();
    benchmark_softmax();
    benchmark_relu();
    benchmark_conv2d();

    std::cout << "\n========================================\n";
    std::cout << "Benchmark complete.\n";

    return 0;
}