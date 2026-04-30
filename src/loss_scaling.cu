#include <cuda_runtime.h>
#include <cstddef>

namespace {

__global__ void scale_gradients_kernel(
    float* gradients, size_t size, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] *= scale;
    }
}

} // namespace

void cuda_scale_gradients(
    float* gradients, size_t size, float scale, cudaStream_t stream) {

    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    scale_gradients_kernel<<<grid_size, block_size, 0, stream>>>(gradients, size, scale);
}