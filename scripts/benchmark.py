#!/usr/bin/env python3
import torch
import time
import numpy as np
from typing import Callable, Dict, List, Tuple

def benchmark_matmul(M: int, N: int, K: int, num_iter: int = 100, warmup: int = 10):
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    for _ in range(warmup):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iter):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iter * 1000
    gflops = (2 * M * N * K) / (avg_time * 1e-3) / 1e9

    return avg_time, gflops

def benchmark_relu(size: int, num_iter: int = 100, warmup: int = 10):
    x = torch.randn(size, device='cuda', dtype=torch.float32)

    for _ in range(warmup):
        torch.relu(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iter):
        torch.relu(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iter * 1000
    return avg_time

def benchmark_conv2d(batch: int, in_ch: int, out_ch: int, H: int, W: int,
                     kernel_size: int, num_iter: int = 100, warmup: int = 10):
    x = torch.randn(batch, in_ch, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_ch, in_ch, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_ch, device='cuda', dtype=torch.float32)

    for _ in range(warmup):
        torch.conv2d(x, weight, bias=bias, padding=kernel_size//2)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iter):
        torch.conv2d(x, weight, bias=bias, padding=kernel_size//2)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iter * 1000
    return avg_time

def benchmark_softmax(batch: int, num_classes: int, num_iter: int = 100, warmup: int = 10):
    x = torch.randn(batch, num_classes, device='cuda', dtype=torch.float32)

    for _ in range(warmup):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iter):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / num_iter * 1000
    return avg_time

def main():
    print("=" * 60)
    print("CUDA Deep Learning Operators Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    print("\n[ MatMul Benchmark (M=N=K=1024) ]")
    time_ms, gflops = benchmark_matmul(1024, 1024, 1024)
    print(f"  Time: {time_ms:.4f} ms")
    print(f"  GFLOPS: {gflops:.2f}")

    print("\n[ ReLU Benchmark (size=10M) ]")
    time_ms = benchmark_relu(10_000_000)
    print(f"  Time: {time_ms:.4f} ms")

    print("\n[ Conv2d Benchmark (N=32, C=64, H=32, W=32, K=3) ]")
    time_ms = benchmark_conv2d(32, 64, 64, 32, 32, 3)
    print(f"  Time: {time_ms:.4f} ms")

    print("\n[ Softmax Benchmark (batch=256, classes=1000) ]")
    time_ms = benchmark_softmax(256, 1000)
    print(f"  Time: {time_ms:.4f} ms")

    print("\n" + "=" * 60)
    print("Benchmark completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
