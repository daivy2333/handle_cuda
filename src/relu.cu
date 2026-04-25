#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void relu_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu_backward_kernel(const float* grad_out, const float* forward_input,
                                      float* grad_in, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = forward_input[idx] > 0.0f ? grad_out[idx] : 0.0f;
    }
}

} // namespace

void cuda_relu(float* data, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    relu_kernel<<<num_blocks, block_size, 0, stream>>>(data, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_relu_backward(const float* grad_out, const float* forward_input,
                         float* grad_in, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    relu_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_out, forward_input, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}
