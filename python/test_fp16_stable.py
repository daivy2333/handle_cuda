"""
FP16 Mixed Precision Training Test - Isolated Models
"""

import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fp32_only(num_iters=50, batch_size=64):
    """Test FP32 training."""
    print("="*60)
    print("FP32 Training Test")
    print("="*60)

    from cuda_ops import CUDAOps
    from model_cuda import SimpleMLP_CUDA

    x_data = np.random.randn(num_iters * batch_size, 784).astype(np.float32) * 0.1
    y_data = np.random.randint(0, 10, num_iters * batch_size).astype(np.int32)
    lr = 0.01

    ops = CUDAOps()
    model_fp32 = SimpleMLP_CUDA(ops)
    x_ptr = ops.alloc(batch_size * 784)

    losses = []
    start = time.time()
    for i in range(num_iters):
        x_batch = x_data[i*batch_size:(i+1)*batch_size]
        y_batch = y_data[i*batch_size:(i+1)*batch_size]
        ops.lib.cuda_memcpy_h2d(x_ptr, x_batch.ctypes.data, x_batch.nbytes)
        logits_ptr = model_fp32.forward(x_ptr, batch_size)
        loss = model_fp32.backward(logits_ptr, y_batch)
        losses.append(loss)
        model_fp32.update(lr)
        if i % 10 == 0:
            print(f"  Iter {i}: loss={loss:.4f}")
    fp32_time = time.time() - start
    ops.free(x_ptr)
    del model_fp32
    del ops

    print(f"  Total time: {fp32_time:.2f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  All losses stable: {all(l < 100 for l in losses)}")
    return fp32_time, losses[-1]


def test_fp16_only(num_iters=50, batch_size=64):
    """Test FP16 mixed precision training."""
    print("\n" + "="*60)
    print("FP16 Mixed Precision Training Test")
    print("="*60)

    from cuda_ops import CUDAOps
    from model_cuda import SimpleMLP_MixedPrecision

    x_data = np.random.randn(num_iters * batch_size, 784).astype(np.float32) * 0.1
    y_data = np.random.randint(0, 10, num_iters * batch_size).astype(np.int32)
    lr = 0.01

    ops = CUDAOps()
    model_fp16 = SimpleMLP_MixedPrecision(ops, loss_scale=128.0)
    x_ptr = ops.alloc(batch_size * 784)

    losses = []
    start = time.time()
    for i in range(num_iters):
        x_batch = x_data[i*batch_size:(i+1)*batch_size]
        y_batch = y_data[i*batch_size:(i+1)*batch_size]
        ops.lib.cuda_memcpy_h2d(x_ptr, x_batch.ctypes.data, x_batch.nbytes)
        logits_ptr = model_fp16.forward(x_ptr, batch_size)
        loss = model_fp16.backward(logits_ptr, y_batch)
        losses.append(loss)
        model_fp16.update(lr)
        if i % 10 == 0:
            print(f"  Iter {i}: loss={loss:.4f}")
    fp16_time = time.time() - start
    ops.free(x_ptr)
    del model_fp16
    del ops

    print(f"  Total time: {fp16_time:.2f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  All losses stable: {all(l < 100 for l in losses)}")
    return fp16_time, losses[-1]


if __name__ == '__main__':
    fp32_time, fp32_loss = test_fp32_only(num_iters=50, batch_size=64)
    fp16_time, fp16_loss = test_fp16_only(num_iters=50, batch_size=64)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  FP32 time: {fp32_time:.2f}s, loss: {fp32_loss:.4f}")
    print(f"  FP16 time: {fp16_time:.2f}s, loss: {fp16_loss:.4f}")
    if fp32_time > 0 and fp16_time > 0:
        print(f"  Speedup: {fp32_time/fp16_time:.2f}x")
    print("="*60)