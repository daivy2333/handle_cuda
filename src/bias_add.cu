#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void bias_add_kernel(const float* input, const float* bias,
                                 float* output, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx < total) {
        size_t col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

__global__ void bias_add_backward_kernel(const float* grad_out,
                                         float* grad_input, float* grad_bias,
                                         size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols) {
        grad_input[idx] = grad_out[idx];
    }

    // Parallel reduction for grad_bias
    if (idx < cols) {
        float sum = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            sum += grad_out[r * cols + idx];
        }
        grad_bias[idx] = sum;
    }
}

} // namespace

void cuda_bias_add(const float* input, const float* bias, float* output,
                    size_t rows, size_t cols, cudaStream_t stream) {
    size_t total = rows * cols;
    int block_size = 256;
    int num_blocks = get_num_blocks(total, block_size);
    bias_add_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, bias, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_bias_add_backward(const float* grad_out, float* grad_input,
                             float* grad_bias, size_t rows, size_t cols,
                             cudaStream_t stream) {
    size_t total = rows * cols;
    int block_size = 256;
    int num_blocks = get_num_blocks(std::max(total, cols), block_size);
    bias_add_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_out, grad_input, grad_bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
