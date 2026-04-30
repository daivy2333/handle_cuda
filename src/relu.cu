#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void relu_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = (val > 0.0f || isnan(val)) ? val : 0.0f;
    }
}

__global__ void relu_vectorized_kernel(float* data, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vec_idx = tid * 4;

    if (vec_idx + 4 <= size) {
        float a, b, c, d;
        load_float4(data + vec_idx, a, b, c, d);
        a = (a > 0.0f || isnan(a)) ? a : 0.0f;
        b = (b > 0.0f || isnan(b)) ? b : 0.0f;
        c = (c > 0.0f || isnan(c)) ? c : 0.0f;
        d = (d > 0.0f || isnan(d)) ? d : 0.0f;
        store_float4(data + vec_idx, a, b, c, d);
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && vec_idx + i < size; ++i) {
            float val = data[vec_idx + i];
            data[vec_idx + i] = (val > 0.0f || isnan(val)) ? val : 0.0f;
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

// Out-of-place ReLU: copies input to output first, then applies ReLU on output
__global__ void relu_out_of_place_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val > 0.0f || isnan(val)) ? val : 0.0f;
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

// Out-of-place ReLU: input is preserved, output gets ReLU result
void cuda_relu_out_of_place(const float* input, float* output, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    relu_out_of_place_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, size);
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
