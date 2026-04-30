"""
Simple Benchmark: FP32 Training Performance
"""

import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuda_ops import CUDAOps
from model_cuda import SimpleMLP_CUDA

def benchmark_simple(num_iters=100, batch_size=64):
    """Quick benchmark without MNIST data."""
    print("="*60)
    print("CUDA MLP Training Benchmark (Synthetic Data)")
    print("="*60)

    # Generate synthetic data
    x_data = np.random.randn(num_iters * batch_size, 784).astype(np.float32)
    y_data = np.random.randint(0, 10, num_iters * batch_size).astype(np.int32)

    lr = 0.01

    # FP32 training
    print("\n=== FP32 Training ===")
    ops = CUDAOps()
    model_fp32 = SimpleMLP_CUDA(ops)
    x_ptr = ops.alloc(batch_size * 784)

    start = time.time()
    for i in range(num_iters):
        x_batch = x_data[i*batch_size:(i+1)*batch_size]
        y_batch = y_data[i*batch_size:(i+1)*batch_size]
        ops.lib.cuda_memcpy_h2d(x_ptr, x_batch.ctypes.data, x_batch.nbytes)
        logits_ptr = model_fp32.forward(x_ptr, batch_size)
        loss = model_fp32.backward(logits_ptr, y_batch)
        model_fp32.update(lr)
    fp32_time = time.time() - start
    ops.free(x_ptr)
    print(f"  Total time: {fp32_time:.2f}s ({num_iters} iterations)")
    print(f"  Time per iteration: {fp32_time/num_iters*1000:.2f}ms")
    print(f"  Throughput: {num_iters * batch_size / fp32_time:.0f} samples/s")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  FP32 Training:  {fp32_time:.2f}s ({fp32_time/num_iters*1000:.2f}ms/iter)")
    print(f"  Throughput:     {num_iters * batch_size / fp32_time:.0f} samples/s")
    print("="*60)
    print("\nNote: FP16 Mixed Precision has precision issues in backward kernel.")
    print("      Currently using FP32 for stable training.")

    return fp32_time

if __name__ == '__main__':
    benchmark_simple(num_iters=100, batch_size=64)