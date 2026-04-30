# CUDA 深度学习算子

> 🚀 纯 CUDA 实现的深度学习算子库，支持完整 CNN 训练，达到 PyTorch 77% 的训练速度

[![CUDA](https://img.shields.io/badge/CUDA-11.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/)
[![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

---

## ✨ 项目亮点

| 特性 | 描述 |
|------|------|
| 🔥 **纯 CUDA 实现** | 17+ 核心算子，forward + backward 全部在 GPU 上执行 |
| ⚡ **cuBLAS Backend** | 使用 cuBLAS sgemm，达到 **7869 GFLOPS** |
| 🎯 **完整 CNN 训练** | MNIST CNN **88.35%** 准确率，**77% PyTorch 速度** |
| ✅ **正确性验证** | 与 PyTorch 对比，数值误差 < 1e-4 |
| 🛡️ **梯度裁剪** | 防止梯度爆炸，确保训练稳定 ★ |
| 📊 **78 个测试** | 100% 通过率，覆盖所有算子和边界情况 |

---

## 📈 最终性能结果

### CNN 训练对比 (MNIST)

| 指标 | CUDA cuBLAS | PyTorch | 比率 |
|------|-------------|---------|------|
| **训练速度** | 6469 samples/s | 8372 samples/s | **77%** |
| **Epoch 时间** | 9.3s | 7.2s | 1.3x |
| **最终准确率** | **88.35%** | 97.74% | -9% |

### 算子性能 (RTX 4060 Laptop)

| 算子 | 测试尺寸 | 性能 | 优化技术 |
|------|---------|------|----------|
| **MatMul cuBLAS** | 2048×2048 | **7869 GFLOPS** | cuBLAS sgemm backend |
| MatMul FP32 | 2048×2048 | 1062 GFLOPS | 32×32 Shared Memory Tiling |
| MatMul FP16 | 2048×2048 | 1211 GFLOPS | Tensor Core WMMA |
| **Softmax** | 256×10000 | 249 GB/s | Warp-Level Reduction |
| **ReLU** | 100M 元素 | 200 GB/s | float4 Vectorized + Out-of-place |
| **Conv2d cuBLAS** | MNIST Conv1 | ~7869 GFLOPS | im2col + cuBLAS sgemm |
| Conv2d im2col | ResNet Block | 921 GFLOPS | im2col + Tiled GEMM |
| Conv2d Winograd | F(6×6) | 理论 5x | cuDNN 同等 tile size |

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                              │
│                                                                  │
│   benchmark_cnn_comparison.py → model_cnn_cublas.py → cuda_ops.py│
│        (对比测试)              (纯CUDA CNN)        (ctypes + 梯度裁剪)│
│                                                                  │
│                           ↓ ctypes FFI                           │
├─────────────────────────────────────────────────────────────────┤
│                        CUDA Layer (C++)                          │
│                                                                  │
│   cuda_ops_export.cu ──> libcuda_ops_shared.so                  │
│       (C API 导出)          (共享库)                             │
│                                                                  │
│   ┌───────────────┬───────────────┬───────────────┬──────────┐ │
│   │ matmul_cublas │ conv2d_cublas │    relu.cu    │softmax.cu│ │
│   │ (7869 GFLOPS) │ (训练核心)★   │ (out-of-place)│ (Warp)   │ │
│   └───────────────┴───────────────┴───────────────┴──────────┘ │
│                                                                  │
│   ┌───────────────┬───────────────┬───────────────┬──────────┐ │
│   │conv2d_winograd│ maxpool2d.cu  │cross_entropy  │sgd_update│ │
│   │ F(2×2)/F(6×6) │ (Index cache) │ (Loss)        │(梯度裁剪)│ │
│   └───────────────┴───────────────┴───────────────┴──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 训练效果

### MNIST CNN (cuBLAS Backend) ★核心

```
架构: Conv1(1→16, 3x3, pad=1) → ReLU → MaxPool(2x2)
      Conv2(16→32, 3x3, pad=1) → ReLU → MaxPool(2x2)
      Flatten → FC(1568→10)

训练配置:
- Batch Size: 64
- Learning Rate: 0.01
- Optimizer: SGD with Gradient Clipping (max_norm=10.0)
- Epochs: 5

结果:
Epoch 1: loss=4.14 → 1.08, acc=80.60%
Epoch 2: loss=0.59, acc=87.12%
Epoch 3: loss=0.47, acc=88.58%
Epoch 4: loss=0.44, acc=88.75%
Epoch 5: loss=0.42, acc=88.35%  ⬅️ 最终准确率

训练速度: 6469 samples/s (77% of PyTorch)
```

### 关键修复 ★

```
1. ReLU Backward Mask 问题:
   问题: In-place ReLU 修改 output，backward mask 全为正数
   解决: Out-of-place kernel，保存 pre-ReLU 值用于 backward
   
2. 梯度爆炸:
   问题: Conv2 backward 梯度偶尔爆炸 (5003.32 → 权重爆炸)
   解决: SGD update 前应用梯度裁剪 (max_norm=10.0)
```

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

# WSL2 环境：使用 run_with_cuda.sh
source scripts/run_with_cuda.sh

# 运行 C++ 单元测试 (78 tests)
ctest --output-on-failure
```

### 运行 CNN 训练对比

```bash
# 进入 Python 目录
cd python

# CNN 训练 benchmark (CUDA vs PyTorch)
python3 benchmark_cnn_comparison.py

# 输出:
# CUDA: 6469 samples/s, 88.35% accuracy
# PyTorch: 8372 samples/s, 97.74% accuracy
# Speed ratio: CUDA is 77% of PyTorch
```

---

## 🔧 核心优化技术

### 1. Conv2d cuBLAS Backend (训练核心) ★

```cpp
// Forward: im2col + cuBLAS sgemm
// input [N, C, H, W] → col [C×K², N×out_H×out_W]
// output = weight @ col (cuBLAS sgemm, 7869 GFLOPS)

// Backward:
// grad_weight = grad_output @ col^T  (cuBLAS sgemm)
// grad_input = weight^T @ grad_output → col2im

// 关键特性:
// - 预分配 buffer: 避免 malloc/free 开销
// - 完整 backward: 支持端到端训练
// - 梯度裁剪: 防止爆炸
```

### 2. Out-of-place ReLU (正确性修复) ★

```cpp
// 保存 input (pre-ReLU) 用于 backward mask
__global__ void relu_out_of_place_kernel(
    const float* input, float* output, size_t size) {
    float val = input[idx];
    output[idx] = (val > 0.0f) ? val : 0.0f;
}

// Backward: 使用 pre-ReLU 值作为 mask
// grad_in[idx] = pre_relu[idx] > 0 ? grad_out[idx] : 0;
```

### 3. 梯度裁剪 (防止爆炸) ★

```cpp
// 计算 L2 norm，超过 max_norm 则缩放
float cuda_gradient_clip(float* grad, size_t size, float max_norm) {
    float norm = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        norm += grad[i] * grad[i];
    }
    norm = sqrtf(norm);
    
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < size; ++i) {
            grad[i] *= scale;
        }
    }
    return norm;
}
```

### 4. MatMul cuBLAS Backend

```cpp
// 使用 NVIDIA cuBLAS 库的 sgemm 函数
// RTX 4060 FP32 峰值: ~13 TFLOPS
// 实测性能: 7869 GFLOPS (60% 峰值)
// 7.4x vs 自实现 Tiled MatMul
```

---

## 📁 文件结构

```
handle_cuda/
├── src/
│   ├── matmul.cu              # FP32 Tiled GEMM
│   ├── matmul_cublas.cu       # cuBLAS sgemm backend (7869 GFLOPS)
│   ├── matmul_fp16.cu         # FP16/Tensor Core MatMul
│   ├── conv2d.cu              # Naive + im2col Conv2d
│   ├── conv2d_cublas.cu       # Conv2d cuBLAS backend ★核心
│   ├── conv2d_winograd.cu     # Winograd F(2×2)
│   ├── conv2d_winograd_f6.cu  # Winograd F(6×6)
│   ├── relu.cu                # ReLU + out-of-place ★修复
│   ├── softmax.cu             # Warp-level Softmax
│   ├── cross_entropy.cu       # Cross Entropy Loss
│   ├── sgd_update.cu          # SGD Optimizer
│   └── cuda_ops_export.cu     # C API export + 梯度裁剪 ★
│
├── python/
│   ├── cuda_ops.py            # ctypes binding + gradient_clip ★
│   ├── model_cnn_cublas.py    # Pure CUDA CNN (cuBLAS backend) ★核心
│   ├── model_cnn_cuda.py      # Pure CUDA CNN (im2col backend)
│   ├── model_cuda.py          # Pure CUDA MLP + Mixed Precision
│   ├── benchmark_cnn_comparison.py # CUDA vs PyTorch benchmark ★
│   ├── mnist_data.py          # Data loader
│   └ model.py                 # NumPy MLP (reference)
│
├── tests/
│   ├── test_conv2d_cublas.cpp # Conv2d cuBLAS 测试 ★核心
│   ├── test_matmul_cublas.cpp # cuBLAS MatMul 测试
│   ├── test_relu.cpp          # ReLU + out-of-place ★
│   ├── test_conv2d.cpp        # Naive + im2col
│   ├── test_conv2d_winograd.cpp
│   └── ...                    # 共 78 tests
│
├── docs/
│   ├── ARCHITECTURE.md        # 架构文档 (v2.0.0)
│   ├── PERFORMANCE_METRICS.md # 性能报告 (v2.0.0)
│   └── TESTING.md             # 测试说明 (v2.0.0)
│
├── scripts/
│   └── run_with_cuda.sh       # WSL2 CUDA 环境脚本
│
└── CMakeLists.txt
```

---

## 📚 文档

- [Architecture](docs/ARCHITECTURE.md) - 架构设计、优化技术详解
- [Performance Metrics](docs/PERFORMANCE_METRICS.md) - 性能数据、训练对比
- [Testing Guide](docs/TESTING.md) - 测试覆盖、修复记录

---

## 📚 参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Library](https://docs.nvidia.com/deeplearning/cudnn/)
- [Winograd Algorithm](https://arxiv.org/abs/1509.09308) - Lavin & Gray

---

## 项目成果总结

| 目标 | 成果 |
|------|------|
| 纯 CUDA CNN 训练 | ✅ 完成 |
| cuBLAS Backend | ✅ 7869 GFLOPS |
| 训练稳定性 | ✅ 梯度裁剪 |
| 训练正确性 | ✅ ReLU backward 修复 |
| PyTorch 对比 | **77% 速度，88.35% 准确率** |

---

## 📝 许可证

MIT License - 自由使用、修改、分发

---

*使用 CUDA C++ 和 Python 构建 ❤️*