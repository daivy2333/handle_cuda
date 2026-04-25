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
