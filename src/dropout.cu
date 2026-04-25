#include "cuda_ops.h"
#include "cuda_util.h"
#include <random>
#include <chrono>
#include <thread>

namespace {

__global__ void dropout_kernel(const float* input, float* output, float* mask,
                               size_t size, float dropout_prob, bool training,
                               unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (training && dropout_prob > 0.0f) {
            unsigned long long state = seed + idx;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            float rand_val = (state * 0x2545F4914F6CDD1Dull >> 32) / 4294967296.0f;

            if (rand_val < dropout_prob || dropout_prob >= 1.0f) {
                output[idx] = 0.0f;
                mask[idx] = 0.0f;
            } else {
                float scale = 1.0f / (1.0f - dropout_prob);
                output[idx] = input[idx] * scale;
                mask[idx] = scale;
            }
        } else {
            output[idx] = input[idx];
            mask[idx] = 1.0f;
        }
    }
}

__global__ void dropout_backward_kernel(const float* grad_out, const float* mask,
                                        float* grad_in, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = grad_out[idx] * mask[idx];
    }
}

unsigned long long generate_seed() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    thread_local std::mt19937_64 rng(static_cast<unsigned long long>(now) ^ thread_id);
    return rng();
}

} // namespace

void cuda_dropout(float* data, float* mask, size_t size, float dropout_prob,
                  bool training, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    unsigned long long seed = training ? generate_seed() : 0;
    dropout_kernel<<<num_blocks, block_size, 0, stream>>>(
        data, data, mask, size, dropout_prob, training, seed);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_dropout(const float* input, float* output, float* mask, size_t size,
                  float dropout_prob, bool training, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    unsigned long long seed = training ? generate_seed() : 0;
    dropout_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, output, mask, size, dropout_prob, training, seed);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_dropout_backward(const float* grad_out, const float* mask,
                           float* grad_in, size_t size, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = get_num_blocks(size, block_size);
    dropout_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_out, mask, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}