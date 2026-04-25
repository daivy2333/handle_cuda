# Performance Results

## Benchmark Configuration

- GPU: NVIDIA GeForce RTX 3080 (or similar)
- CUDA: 11.5.119
- Test date: 2026-04-25

## Results

### MatMul

| Size | Time (ms) | GFLOPS |
|------|-----------|--------|
| 512x512 | 0.284 | 944 |
| 1024x1024 | 2.10 | 1023 |
| 2048x2048 | 15.88 | 1082 |

### Softmax

| Batch | Classes | Time (ms) | Bandwidth (GB/s) |
|-------|---------|-----------|------------------|
| 256 | 100 | 0.0155 | 13.2 |
| 256 | 1000 | 0.00998 | 205.3 |
| 256 | 10000 | 0.0842 | 243.3 |

### ReLU

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 1M | 0.014 | 596 |
| 10M | 0.42 | 199 |
| 100M | 4.18 | 201 |

### Conv2d

| Config | Time (ms) | GFLOPS |
|--------|-----------|--------|
| N=32, C=64, H=W=32, out_C=64, K=3 | 2.78 | 765 |
| N=16, C=128, H=W=16, out_C=128, K=3 | 1.00 | 927 |
| N=1, C=3, H=W=224, out_C=64, K=7 | 1.29 | 692 |

## Comparison with PyTorch

Run `python3 scripts/benchmark.py` for PyTorch comparison.

## Optimization Techniques Used

1. **MatMul**: 32x32 shared memory tiling for efficient data reuse
2. **Softmax**: Warp-level reduction using shuffle instructions (__shfl_down_sync)
3. **ReLU**: float4 vectorized memory access for bandwidth efficiency
4. **Conv2d**: im2col transformation + tiled GEMM