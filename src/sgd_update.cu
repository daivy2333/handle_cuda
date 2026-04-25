#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void sgd_update_kernel(float* param, const float* grad,
                                   size_t size, float learning_rate) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= learning_rate * grad[idx];
    }
}

} // namespace

extern "C" {

void cuda_sgd_update(float* param, const float* grad,
                      size_t size, float learning_rate,
                      cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    sgd_update_kernel<<<num_blocks, block_size, 0, stream>>>(
        param, grad, size, learning_rate);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"