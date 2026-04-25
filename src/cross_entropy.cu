#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

__global__ void cross_entropy_forward_kernel(const float* logits, const int* targets,
                                              float* loss, float* grad_logits,
                                              size_t batch_size, size_t num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        const float* logits_row = logits + batch_idx * num_classes;
        int target = targets[batch_idx];

        // Find max for numerical stability
        float max_val = logits_row[0];
        for (int j = 1; j < num_classes; ++j) {
            max_val = fmaxf(max_val, logits_row[j]);
        }

        // Compute softmax denominator
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum += expf(logits_row[j] - max_val);
        }

        // Compute loss contribution: -log(softmax[target])
        float log_softmax_target = logits_row[target] - max_val - logf(sum);

        // Use atomic add for batch-level loss accumulation
        atomicAdd(loss, -log_softmax_target / batch_size);

        // Compute gradient: (softmax - one_hot) / batch_size
        if (grad_logits != nullptr) {
            float* grad_row = grad_logits + batch_idx * num_classes;
            for (int j = 0; j < num_classes; ++j) {
                float softmax_val = expf(logits_row[j] - max_val) / sum;
                grad_row[j] = (softmax_val - (j == target ? 1.0f : 0.0f)) / batch_size;
            }
        }
    }
}

} // namespace

extern "C" {

void cuda_cross_entropy_loss(const float* logits, const int* targets,
                              float* loss, float* grad_logits,
                              size_t batch_size, size_t num_classes,
                              cudaStream_t stream) {
    // Initialize loss to 0
    CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), stream));

    int block_size = 256;
    int num_blocks = get_num_blocks(batch_size, block_size);

    cross_entropy_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        logits, targets, loss, grad_logits, batch_size, num_classes);

    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"