# CUDA 深度学习算子

> 🚀 纯 CUDA 实现的深度学习算子库，从零构建，支持完整神经网络训练

[![CUDA](https://img.shields.io/badge/CUDA-11.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/)
[![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

---

## ✨ 项目亮点

| 特性 | 描述 |
|------|------|
| 🔥 **纯 CUDA 实现** | 12 个核心算子，forward + backward 全部在 GPU 上执行 |
| ⚡ **高性能优化** | Shared Memory Tiling、Warp-Level Reduction、im2col + GEMM |
| 🎯 **完整训练流程** | MNIST MLP **95.36%** / CNN **97.92%** 准确率 |
| ✅ **正确性验证** | 与 PyTorch 对比，数值误差 < 1e-6 |
| 📊 **59 个测试** | 100% 通过率，覆盖所有算子和边界情况 |

---

## 📈 性能数据

> 测试硬件：**Tesla T4 (16GB)** · FP32 峰值 8.1 TFLOPS · 显存带宽 320 GB/s

### 算子性能

| 算子 | 测试尺寸 | 性能 | 优化技术 | vs Naive |
|------|---------|------|----------|----------|
| **MatMul** | 2048×2048 | 1062 GFLOPS (13% 峰值) | 32×32 Shared Memory Tiling | +26% |
| **Softmax** | 256×10000 | 249 GB/s (78% 带宽) | Warp-Level Reduction (`__shfl_down_sync`) | **17.8x** |
| **ReLU** | 100M 元素 | 200 GB/s (63% 带宽) | float4 Vectorized Memory Access | **4x** |
| **Conv2d** | 16×128×16×16, K=3 | 921 GFLOPS (11% 峰值) | im2col + Tiled GEMM | **357x** |

### 训练性能对比

![Performance Comparison](docs/performance_comparison.png)

| 实现 | Forward+Backward | vs NumPy |
|------|------------------|----------|
| NumPy (CPU) | 22.58 ms/batch | 1.0x (baseline) |
| **Pure CUDA** | 2.13 ms/batch | **10.61x faster** |
| PyTorch (est.) | ~3.2 ms/batch | ~7x faster |

> 💡 小模型 (784→256→128→10) 上，纯 CUDA 消除 CPU-GPU 数据传输瓶颈后，性能超越 NumPy **10 倍以上**

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                              │
│                                                                  │
│   train_mnist_cuda.py ──> model_cuda.py ──> cuda_ops.py         │
│        (训练循环)          (纯CUDA MLP)      (ctypes binding)    │
│                                                                  │
│                           ↓ ctypes FFI                           │
├─────────────────────────────────────────────────────────────────┤
│                        CUDA Layer (C++)                          │
│                                                                  │
│   cuda_ops_export.cu ──> libcuda_ops_shared.so                  │
│       (C API 导出)          (共享库)                             │
│                                                                  │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│   │  matmul.cu  │  relu.cu    │  softmax.cu │  conv2d.cu  │     │
│   │  (Tiled)    │  (Vector)   │  (Warp)     │  (im2col)   │     │
│   └─────────────┴─────────────┴─────────────┴─────────────┘     │
│                                                                  │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│   │ bias_add.cu │ maxpool2d.cu│cross_entropy│ sgd_update  │     │
│   │ (Broadcast) │ (Pooling)   │  (Loss)     │ (Optimizer) │     │
│   └─────────────┴─────────────┴─────────────┴─────────────┘     │
│                                                                  │
│                          flatten.cu (Reshape)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 训练效果

### MNIST MLP (3-layer)

```
Layer: 784 → 256 (ReLU) → 128 (ReLU) → 10 (Softmax)

Epoch  1: loss=0.7472, acc=89.37%
Epoch  5: loss=0.2245, acc=93.58%
Epoch 10: loss=0.1550, acc=95.36%  ⬅️ 最终准确率

训练时间: 8.24 秒 (10 epochs, batch_size=64)
```

### MNIST CNN (2-Conv)

```
Layer: Conv1(1→16, 3x3) → ReLU → MaxPool(2x2)
       Conv2(16→32, 3x3) → ReLU → MaxPool(2x2)
       Flatten → FC(1568→10)

Epoch  1: loss=0.47, acc=91.31%
Epoch  5: loss=0.13, acc=96.75%
Epoch 10: loss=0.07, acc=97.92%  ⬅️ 最终准确率

训练时间: 11分36秒 (10 epochs, batch_size=64)
速度: ~1000 samples/s
```

### 与 PyTorch 对比

| 模型 | 纯 CUDA | PyTorch (GPU) | 差距 |
|------|---------|---------------|------|
| MLP | 95.36% (8.24s) | ~95% (est.) | 相当 |
| CNN | 97.92% (11m36s) | 97.34% (14.8s) | CUDA准确率+0.58%，速度差42x |

> 💡 速度差距源于 PyTorch 使用 cuDNN/Tensor Core，我们的实现是纯 FP32 教育性实现

---

## 🚀 快速开始

### 构建 CUDA 库

```bash
# 克隆项目
git clone https://github.com/yourname/handle_cuda.git
cd handle_cuda

# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行 C++ 单元测试 (59 tests)
ctest --output-on-failure
```

### 运行 MNIST 训练

```bash
# 进入 Python 目录
cd python

# MLP 训练 (95.36% accuracy, 8.24s)
python train_mnist_cuda.py

# CNN 训练 (97.92% accuracy, 11m36s)
python train_mnist_cnn_cuda.py

# PyTorch 对比 (97.34% accuracy, 14.8s)
python train_mnist_cnn_pytorch.py
```

---

## 🔧 优化技术详解

### 1. MatMul: 32×32 Shared Memory Tiling

```cpp
// 每个 thread block 处理 32×32 的输出 tile
__shared__ float As[32][32];  // A 的 tile
__shared__ float Bs[32][32];  // B 的 tile

// 减少全局内存访问: K 次 → K/32 次
// 性能: 1062 GFLOPS @ 2048×2048 (Tesla T4, 13% 峰值)
// Naive kernel: ~760 GFLOPS (未使用 shared memory，每线程独立计算)
```

### 2. Softmax: Warp-Level Reduction

```cpp
// 使用 shuffle 指令进行 warp 内 reduction
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 每个 warp (32 threads) 处理一个 batch
// 性能: 249 GB/s @ 256×10000 (Tesla T4, 78% 显存带宽利用率)
// Naive kernel: 14 GB/s (每线程串行计算 max/sum，无并行 reduction)
```

### 3. Conv2d: im2col + Tiled GEMM

```cpp
// im2col: 将卷积转换为矩阵乘法
// input [N, C, H, W] → col [C×K², N×out_H×out_W]

// 复用优化后的 MatMul kernel
// 性能: 921 GFLOPS @ 16×128×16×16, K=3 (Tesla T4)

// Naive kernel 特征 (conv2d.cu:84-128):
// - 每个 block (256 threads) 只计算一个输出像素
// - 6层嵌套循环：in_c × kernel_h × kernel_w
// - 无 shared memory：input/weight 无法复用
// - 内存访问不合并：相邻线程访问完全不同地址
// → 仅 2.58 GFLOPS，差距 357x（架构瓶颈，非 bug）
```

---

## 📁 文件结构

```
handle_cuda/
├── src/
│   ├── matmul.cu           # Tiled GEMM (1062 GFLOPS)
│   ├── relu.cu             # Vectorized ReLU
│   ├── softmax.cu          # Warp-level Softmax
│   ├── conv2d.cu           # im2col + GEMM + Backward optimization
│   ├── bias_add.cu         # Broadcasting
│   ├── maxpool2d.cu        # Pooling
│   ├── cross_entropy.cu    # Loss function
│   ├── sgd_update.cu       # SGD optimizer
│   ├── flatten.cu          # Reshape
│   ├── sigmoid.cu          # Sigmoid activation
│   ├── tanh.cu             # Tanh activation
│   ├── dropout.cu          # Dropout regularization
│   └── cuda_ops_export.cu  # C API 导出
│
├── include/
│   └── cuda_ops.h          # 头文件 + Conv2dBackwardBuffers struct
│
├── python/
│   ├── cuda_ops.py         # ctypes binding (含预分配缓冲区API)
│   ├── model_cuda.py       # 纯 CUDA MLP
│   ├── model_cnn_cuda.py   # 纯 CUDA CNN (预分配优化)
│   ├── model.py            # NumPy MLP (对比)
│   ├── train_mnist_cuda.py # MLP 训练脚本
│   ├── train_mnist_cnn_cuda.py # CNN 训练脚本
│   ├── train_mnist_cnn_pytorch.py # PyTorch 对比
│   ├── mnist_data.py       # 数据加载
│   └ performance_comparison.py # MLP 性能对比
│
├── tests/
│   ├── test_matmul.cpp     # 5 tests
│   ├── test_relu.cpp       # 5 tests
│   ├── test_softmax.cpp    # 4 tests
│   ├── test_conv2d.cpp     # 6 tests
│   ├── test_edge_cases.cpp # 9 tests (边界情况)
│   └ ...                   # 共 59 tests
│
├── docs/
│   ├── PROJECT_PLAN.md     # 项目计划
│   ├── PERFORMANCE_METRICS.md  # 性能报告 + CNN对比分析
│   └ performance_comparison.png # MLP 可视化图
│
└── CMakeLists.txt
```

---

## 📚 参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-best-practices-guide/)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Library](https://docs.nvidia.com/deeplearning/cudnn/)
- [PyTorch ATen Native CUDA](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda)

---

## 下一步

### ✅ 已完成

| 功能 | 状态 | 成果 |
|------|------|------|
| MLP 训练 | ✅ 完成 | 95.36% 准确率，10.61x vs NumPy |
| CNN 训练 | ✅ 完成 | 97.92% 准确率，im2col+GEMM优化 |
| 边界情况测试 | ✅ 完成 | 9 个新增测试，100% 通过 |
| 预分配缓冲区优化 | ✅ 完成 | 消除 malloc/free，+27% 速度 |

### 进一步扩展方向

| 方向 | 描述 | 预期收益 |
|------|------|---------|
| **cuBLAS 集成** | 使用 cuBLAS sgemm 替代自实现 matmul | 速度 +2-3x |
| **cuDNN 集成** | 使用 cuDNN conv/pool kernels | CNN 速度 +10-20x |
| **FP16/Tensor Core** | 修改 kernel 支持 `half` 精度 | 推理速度 +2-4x |
| **BatchNorm/LayerNorm** | 添加归一化算子 | 支持更深的网络 |
| **Attention** | Transformer 核心算子 | 支持 LLM 架构 |
| **Winograd 算法** | 手动实现 3x3 Winograd | Conv2d +2.5x |

### 与其他项目的关联

本项目与 [riscv-bsv-processor](https://github.com/yourname/riscv-bsv-processor) 共同探索底层硬件加速：
- CUDA 的 warp-level reduction 思路可借鉴到 RISC-V V 扩展（RVV）
- 两者都关注：内存访问优化、并行归约、分块计算

---

## 📝 许可证

MIT License - 自由使用、修改、分发

---

*使用 CUDA C++ 和 Python 构建 ❤️*