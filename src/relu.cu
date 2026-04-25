#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void relu_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu_vectorized_kernel(float* data, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vec_idx = tid * 4;

    if (vec_idx + 4 <= size) {
        float a, b, c, d;
        load_float4(data + vec_idx, a, b, c, d);
        a = fmaxf(0.0f, a);
        b = fmaxf(0.0f, b);
        c = fmaxf(0.0f, c);
        d = fmaxf(0.0f, d);
        store_float4(data + vec_idx, a, b, c, d);
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && vec_idx + i < size; ++i) {
            data[vec_idx + i] = fmaxf(0.0f, data[vec_idx + i]);
        }
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
    // Vectorized: 4 elements per thread
    int block_size = 256;
    int num_blocks = (size / 4 + block_size - 1) / block_size;
    if (num_blocks == 0) num_blocks = 1;

    relu_vectorized_kernel<<<num_blocks, block_size, 0, stream>>>(data, size);
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
