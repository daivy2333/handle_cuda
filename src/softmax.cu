#include "cuda_ops.h"
#include "cuda_util.h"
#include <cmath>

namespace {

// Warp-level reduction helpers
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// Naive kernel: one thread per batch (original implementation)
__global__ void softmax_kernel(const float* input, float* output,
                                size_t batch_size, size_t num_classes) {
    size_t batch_idx = blockIdx.x;

    if (batch_idx < batch_size) {
        const float* input_row = input + batch_idx * num_classes;
        float* output_row = output + batch_idx * num_classes;

        float max_val = -INFINITY;
        for (size_t i = 0; i < num_classes; ++i) {
            max_val = fmaxf(max_val, input_row[i]);
        }

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            sum += expf(input_row[i] - max_val);
        }

        for (size_t i = 0; i < num_classes; ++i) {
            output_row[i] = expf(input_row[i] - max_val) / sum;
        }
    }
}

// Warp-level kernel: one warp (32 threads) per batch
__global__ void softmax_warp_kernel(const float* input, float* output,
                                    size_t batch_size, size_t num_classes) {
    // Each warp handles one batch
    int batch_idx = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (batch_idx >= batch_size) return;

    const float* input_row = input + batch_idx * num_classes;
    float* output_row = output + batch_idx * num_classes;

    // Find max (distributed across warp)
    float max_val = -INFINITY;
    for (int i = lane; i < num_classes; i += 32) {
        max_val = fmaxf(max_val, input_row[i]);
    }
    max_val = warp_reduce_max(max_val);

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = lane; i < num_classes; i += 32) {
        sum += expf(input_row[i] - max_val);
    }
    sum = warp_reduce_sum(sum);

    // Write output
    for (int i = lane; i < num_classes; i += 32) {
        output_row[i] = expf(input_row[i] - max_val) / sum;
    }
}

__global__ void softmax_backward_kernel(const float* grad_out, const float* forward_output,
                                         float* grad_in, size_t batch_size, size_t num_classes) {
    size_t batch_idx = blockIdx.x;

    if (batch_idx < batch_size) {
        const float* go_row = grad_out + batch_idx * num_classes;
        const float* fo_row = forward_output + batch_idx * num_classes;
        float* gi_row = grad_in + batch_idx * num_classes;

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            sum += go_row[i] * fo_row[i];
        }

        for (size_t i = 0; i < num_classes; ++i) {
            gi_row[i] = fo_row[i] * (go_row[i] - sum);
        }
    }
}

} // namespace

void cuda_softmax(const float* input, float* output, size_t batch_size,
                   size_t num_classes, cudaStream_t stream) {
    // Use warp-level kernel: 4 warps per block = 128 threads per block
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    softmax_warp_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input, output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_softmax_backward(const float* grad_out, const float* forward_output,
                             float* grad_in, size_t batch_size,
                             size_t num_classes, cudaStream_t stream) {
    int num_blocks = batch_size;
    softmax_backward_kernel<<<num_blocks, 1, 0, stream>>>(
        grad_out, forward_output, grad_in, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}
