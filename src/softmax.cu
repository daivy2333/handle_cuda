#include "cuda_ops.h"
#include "cuda_util.h"
#include <cmath>

namespace {

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
    int num_blocks = batch_size;
    softmax_kernel<<<num_blocks, 1, 0, stream>>>(
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
