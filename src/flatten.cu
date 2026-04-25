#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void flatten_kernel(const float* input, float* output,
                                size_t batch, size_t C, size_t H, size_t W) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * C * H * W;

    if (idx < total) {
        output[idx] = input[idx];
    }
}

__global__ void flatten_backward_kernel(const float* grad_flat, float* grad_input,
                                         size_t batch, size_t C, size_t H, size_t W) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * C * H * W;

    if (idx < total) {
        grad_input[idx] = grad_flat[idx];
    }
}

} // namespace

extern "C" {

void cuda_flatten(const float* input, float* output,
                   size_t batch, size_t C, size_t H, size_t W,
                   cudaStream_t stream) {
    size_t total = batch * C * H * W;
    int block_size = 256;
    int num_blocks = get_num_blocks(total, block_size);
    flatten_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, batch, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_flatten_backward(const float* grad_flat, float* grad_input,
                            size_t batch, size_t C, size_t H, size_t W,
                            cudaStream_t stream) {
    size_t total = batch * C * H * W;
    int block_size = 256;
    int num_blocks = get_num_blocks(total, block_size);
    flatten_backward_kernel<<<num_blocks, block_size, 0, stream>>>(grad_flat, grad_input, batch, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"