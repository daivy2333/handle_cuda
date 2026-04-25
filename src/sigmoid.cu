#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void sigmoid_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_out,
                                         const float* forward_output,
                                         float* grad_in, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = grad_out[idx] * forward_output[idx] * (1.0f - forward_output[idx]);
    }
}

} // namespace

void cuda_sigmoid(float* data, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    sigmoid_kernel<<<num_blocks, block_size, 0, stream>>>(data, data, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sigmoid(const float* input, float* output, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    sigmoid_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sigmoid_backward(const float* grad_out, const float* forward_output,
                           float* grad_in, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    sigmoid_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_out, forward_output, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}