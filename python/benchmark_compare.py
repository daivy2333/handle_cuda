"""
Performance Comparison Script for CNN Training Optimizations
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cuda_ops import CUDAOps
    HAS_CUDA_OPS = True
except Exception as e:
    print(f"Warning: Could not load cuda_ops: {e}")
    HAS_CUDA_OPS = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch not available")
    HAS_TORCH = False


def benchmark_im2col(ops, N, C, H, W, out_C, kernel=3, stride=1, pad=1, warmup=10, iterations=50):
    """Benchmark im2col Conv2d implementation"""
    if not HAS_CUDA_OPS:
        return None

    out_H = H - 2 * pad - kernel + 1
    out_W = W - 2 * pad - kernel + 1

    # Pre-allocate buffers
    d_input = ops.alloc(N * C * H * W)
    d_weight = ops.alloc(out_C * C * kernel * kernel)
    d_output = ops.alloc(N * out_C * out_H * out_W)

    # Allocate im2col buffers
    col_rows = C * kernel * kernel
    col_cols = N * out_H * out_W
    col_buffer = ops.alloc(col_rows * col_cols)
    gemm_buffer = ops.alloc(out_C * col_cols)

    # Generate random data
    input_np = np.random.randn(N, C, H, W).astype(np.float32)
    weight_np = np.random.randn(out_C, C, kernel, kernel).astype(np.float32)

    ops.lib.cuda_memcpy_h2d(d_input, input_np.ctypes.data, input_np.nbytes)
    ops.lib.cuda_memcpy_h2d(d_weight, weight_np.ctypes.data, weight_np.nbytes)
    ops.sync()

    # Warmup
    for _ in range(warmup):
        ops.conv2d_im2col(d_input, d_weight, None, N, C, H, W, out_C, kernel, kernel, stride, stride, pad, pad)
        ops.sync()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        ops.conv2d_im2col(d_input, d_weight, None, N, C, H, W, out_C, kernel, kernel, stride, stride, pad, pad)
        ops.sync()
    elapsed = time.time() - start

    ops.free(d_input)
    ops.free(d_weight)
    ops.free(d_output)
    ops.free(col_buffer)
    ops.free(gemm_buffer)

    return {
        'time_ms': (elapsed / iterations) * 1000,
        'samples_per_sec': N / (elapsed / iterations)
    }


def benchmark_pytorch(N, C, H, W, out_C, kernel=3, stride=1, pad=1, warmup=10, iterations=50):
    """Benchmark PyTorch Conv2d"""
    if not HAS_TORCH:
        return None
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    input_t = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    weight_t = torch.randn(out_C, C, kernel, kernel, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = F.conv2d(input_t, weight_t, padding=pad, stride=stride)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = F.conv2d(input_t, weight_t, padding=pad, stride=stride)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return {
        'time_ms': (elapsed / iterations) * 1000,
        'samples_per_sec': N / (elapsed / iterations)
    }


def benchmark_matmul(ops, M, N, K, warmup=10, iterations=100):
    """Benchmark matmul operation"""
    if not HAS_CUDA_OPS:
        return None
    
    d_A = ops.alloc(M * K)
    d_B = ops.alloc(K * N)
    d_C = ops.alloc(M * N)
    
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    
    ops.lib.cuda_memcpy_h2d(d_A, A_np.ctypes.data, A_np.nbytes)
    ops.lib.cuda_memcpy_h2d(d_B, B_np.ctypes.data, B_np.nbytes)
    ops.sync()
    
    # Warmup
    for _ in range(warmup):
        ops.matmul(d_A, d_B, M, N, K)
        ops.sync()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        ops.matmul(d_A, d_B, M, N, K)
        ops.sync()
    elapsed = time.time() - start
    
    ops.free(d_A)
    ops.free(d_B)
    ops.free(d_C)
    
    flops = 2 * M * N * K
    gflops = flops / (elapsed / iterations) / 1e6
    
    return {
        'time_ms': (elapsed / iterations) * 1000,
        'gflops': gflops
    }


def run_conv2d_comparison():
    """Run Conv2d performance comparison"""
    print("\n" + "="*70)
    print("Conv2d Performance Comparison")
    print("="*70)
    
    configs = [
        {'N': 64, 'C': 16, 'H': 28, 'W': 28, 'out_C': 32, 'name': 'MNIST Layer1'},
        {'N': 64, 'C': 32, 'H': 14, 'W': 14, 'out_C': 64, 'name': 'MNIST Layer2'},
        {'N': 32, 'C': 64, 'H': 32, 'W': 32, 'out_C': 64, 'name': 'ResNet Block'},
    ]
    
    for cfg in configs:
        name = cfg.pop('name')  # Extract name before passing to functions
        print(f"\n--- {name} ---")
        print(f"    N={cfg['N']}, C={cfg['C']}, H={cfg['H']}, W={cfg['W']}, out_C={cfg['out_C']}")
        
        if HAS_CUDA_OPS:
            ops = CUDAOps()
            result = benchmark_im2col(ops, **cfg)
            if result:
                print(f"  im2col:   {result['time_ms']:.3f} ms/batch, {result['samples_per_sec']:.0f} samples/s")
        
        if HAS_TORCH:
            result = benchmark_pytorch(**cfg)
            if result:
                print(f"  PyTorch:  {result['time_ms']:.3f} ms/batch, {result['samples_per_sec']:.0f} samples/s")
        
        cfg['name'] = name  # Put name back


def run_matmul_comparison():
    """Run Matmul performance comparison"""
    print("\n" + "="*70)
    print("MatMul Performance Comparison")
    print("="*70)
    
    configs = [
        {'M': 512, 'N': 512, 'K': 512, 'name': '512x512x512'},
        {'M': 1024, 'N': 1024, 'K': 1024, 'name': '1024x1024x1024'},
        {'M': 2048, 'N': 2048, 'K': 2048, 'name': '2048x2048x2048'},
    ]
    
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\n--- {name} ---")
        
        if HAS_CUDA_OPS:
            ops = CUDAOps()
            result = benchmark_matmul(ops, **cfg)
            if result:
                print(f"  CUDA:     {result['time_ms']:.3f} ms, {result['gflops']:.1f} GFLOPS")
        
        cfg['name'] = name


def run_training_simulation(batch_size=64):
    """Simulate training throughput"""
    print("\n" + "="*70)
    print(f"Training Simulation (batch_size={batch_size})")
    print("="*70)
    
    if not HAS_CUDA_OPS or not HAS_TORCH:
        print("Skipping - cuda_ops or PyTorch not available")
        return
    
    ops = CUDAOps()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = batch_size
    
    print("\n--- Conv1: N=64, C=1, H=28, W=28, out_C=16 ---")
    
    d_input = ops.alloc(N * 1 * 28 * 28)
    input_np = np.random.randn(N, 1, 28, 28).astype(np.float32)
    ops.lib.cuda_memcpy_h2d(d_input, input_np.ctypes.data, input_np.nbytes)
    
    d_w1 = ops.alloc(16 * 1 * 3 * 3)
    w1_np = np.random.randn(16, 1, 3, 3).astype(np.float32)
    ops.lib.cuda_memcpy_h2d(d_w1, w1_np.ctypes.data, w1_np.nbytes)
    ops.sync()
    
    iterations = 30
    t0 = time.time()
    for _ in range(iterations):
        out1 = ops.conv2d(d_input, d_w1, None, N, 1, 28, 28, 16, 3, 3, 1, 1, 1, 1)
        ops.sync()
    t1 = time.time()
    
    cuda_time = (t1 - t0) / iterations
    print(f"  CUDA im2col: {cuda_time*1000:.3f} ms/batch, {N/cuda_time:.0f} samples/s")
    
    # PyTorch
    x = torch.randn(N, 1, 28, 28, device=device)
    w = torch.randn(16, 1, 3, 3, device=device)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iterations):
        y = F.conv2d(x, w, padding=1, stride=1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    
    torch_time = (t1 - t0) / iterations
    print(f"  PyTorch:     {torch_time*1000:.3f} ms/batch, {N/torch_time:.0f} samples/s")
    print(f"  PyTorch is {torch_time/cuda_time:.1f}x faster")
    
    ops.free(d_input)
    ops.free(d_w1)


def main():
    print("="*70)
    print("CNN Training Performance Benchmark")
    print("="*70)
    
    if not HAS_CUDA_OPS:
        print("\nWarning: cuda_ops library not available")
    if not HAS_TORCH:
        print("Warning: PyTorch not available")
    
    run_conv2d_comparison()
    run_matmul_comparison()
    run_training_simulation(batch_size=64)


if __name__ == '__main__':
    main()
