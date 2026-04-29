# Architecture

## 项目结构

```
handle_cuda/
├── src/                   # CUDA kernels (.cu)
│   ├── matmul.cu          # Tiled GEMM (1062 GFLOPS)
│   ├── relu.cu            # Vectorized ReLU
│   ├── softmax.cu         # Warp-level Softmax
│   ├── bias_add.cu        # Broadcasting
│   ├── conv2d.cu          # im2col + GEMM
│   ├── maxpool2d.cu       # Pooling
│   ├── sigmoid.cu         # Sigmoid activation
│   ├── tanh.cu            # Tanh activation
│   ├── dropout.cu         # Dropout
│   ├── cross_entropy.cu   # Cross entropy loss
│   ├── sgd_update.cu      # SGD optimizer
│   ├── flatten.cu         # Reshape
│   └── cuda_ops_export.cu # C API export
│
├── include/
│   ├── cuda_ops.h         # Public API
│   └── cuda_util.h        # Internal utilities
│
├── python/
│   ├── cuda_ops.py        # ctypes binding
│   ├── model_cuda.py      # Pure CUDA MLP
│   ├── model.py           # NumPy MLP (reference)
│   ├── train_mnist_cuda.py# Training script
│   ├── performance_comparison.py
│   └── mnist_data.py      # Data loader
│
├── tests/                 # GoogleTest (59 tests, 100% pass)
│   ├── test_edge_cases.cpp # Boundary tests (9 tests)
│   └── ...                 # Operator tests
│
└── docs/
    ├── PERFORMANCE_METRICS.md
    ├── ARCHITECTURE.md
    ├── TESTING.md
    └── CUDA_GUIDE.md
```

## 已实现算子

| 算子 | Forward | Backward | 优化技术 | 性能 |
|------|---------|----------|----------|------|
| **MatMul** | ✅ | ✅ | 32×32 Shared Memory Tiling | 1062 GFLOPS |
| **ReLU** | ✅ | ✅ | float4 Vectorization | 200 GB/s |
| **Softmax** | ✅ | ✅ | Warp-Level Reduction | 249 GB/s |
| **BiasAdd** | ✅ | ✅ | Broadcasting | - |
| **Conv2d** | ✅ | ✅ | im2col + Tiled GEMM | 921 GFLOPS |
| **MaxPool2d** | ✅ | ✅ | - | - |
| **Sigmoid** | ✅ | ✅ | - | - |
| **Tanh** | ✅ | ✅ | - | - |
| **Dropout** | ✅ | ✅ | - | - |
| **CrossEntropy** | ✅ | ✅ | Numerical stability | - |
| **SGD Update** | ✅ | - | - | - |
| **Flatten** | ✅ | ✅ | Memory copy | - |

## 优化技术

### MatMul: Shared Memory Tiling

```
每个 thread block 处理 32×32 输出 tile

┌─────────────────────────────────────┐
│  A tile (32×32)    B tile (32×32)    │
│  ┌──────────────┐  ┌──────────────┐  │
│  │ Shared Mem   │  │ Shared Mem   │  │
│  │ 32×32 floats │  │ 32×32 floats │  │
│  └──────────────┘  └──────────────┘  │
│                                      │
│  每个 thread 计算 C[row][col]        │
│  遍历 K/32 个 tiles                  │
└─────────────────────────────────────┘

效果: 全局内存访问减少 32 倍
性能: 1062 GFLOPS (2048×2048)
```

### Softmax: Warp-Level Reduction

```cpp
// 使用 shuffle 指令在 warp 内做 reduction
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 每个 warp (32 threads) 处理一个 batch
效果: 串行 max/sum → 并行 warp reduction
性能: 249 GB/s, 17.8x vs naive
```

### Conv2d: im2col + GEMM

```
input [N, C, H, W] → im2col → col [C×K², N×out_H×out_W]
                               ↓
                     MatMul (复用优化后的 kernel)
                               ↓
                     output [N, out_C, out_H, out_W]

效果: 复用优化后的 MatMul kernel
性能: 921 GFLOPS, 281x vs naive
```

### ReLU: Vectorized Memory Access

```cpp
// 使用 float4 一次读写 4 个元素
float4 data = *reinterpret_cast<float4*>(ptr);

效果: 更好的内存带宽利用
性能: 200 GB/s, 4x vs naive
```

## 设计原则

1. **单算子单文件** - 每个算子一个 `.cu` 文件，便于维护
2. **零外部依赖** - 仅依赖 CUDA Toolkit 和 GoogleTest
3. **测试驱动** - 59 个单元测试，覆盖 forward/backward + 边界场景
4. **可验证** - 与 PyTorch 数值对比，误差 < 1e-6
5. **边界覆盖** - NaN/Inf、显存压力、batch=1、非方阵矩阵