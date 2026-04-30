# Testing Guide

## 测试框架

使用 GoogleTest 作为单元测试框架。

## 运行测试

```bash
# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# WSL2 环境：使用 run_with_cuda.sh
source scripts/run_with_cuda.sh
ctest --output-on-failure

# 运行所有测试 (78 tests)
ctest --output-on-failure

# 运行单个测试
./bin/test_matmul
./bin/test_matmul_cublas      # cuBLAS 后端测试
./bin/test_conv2d
./bin/test_conv2d_cublas      # Conv2d cuBLAS 测试 ★核心
./bin/test_conv2d_winograd    # Winograd F(2×2)
./bin/test_conv2d_winograd_f6 # Winograd F(6×6)
./bin/test_fp16_tensor_core   # Tensor Core 测试
./bin/test_conv2d_fused       # Kernel Fusion 测试
./bin/test_edge_cases         # 边界测试
./bin/test_relu               # ReLU 测试 (含 out-of-place)
./bin/test_softmax            # Softmax 测试
```

## 测试覆盖

| 算子 | Forward | Backward | 边界测试 | 性能测试 |
|------|---------|----------|----------|----------|
| MatMul FP32 | ✅ | ✅ | 转置, 大矩阵 | ✅ |
| MatMul cuBLAS | ✅ | ✅ | 与自实现对比 | ✅ |
| Conv2d im2col | ✅ | ✅ | im2col 验证 | ✅ |
| **Conv2d cuBLAS** | ✅ | ✅ | forward/backward 正确性 | ✅ ★核心 |
| Conv2d Winograd F(2×2) | ✅ | - | im2col 对比 | ✅ |
| Conv2d Winograd F(6×6) | ✅ | - | im2col 对比, 多通道 | ✅ |
| Conv+ReLU Fused | ✅ | ✅ | 分离 kernel 对比 | ✅ |
| ReLU (out-of-place) | ✅ | ✅ | NaN/Inf, mask正确性 | ✅ ★修复 |
| Softmax | ✅ | ✅ | 和为1, NaN/Inf | ✅ |
| BiasAdd | ✅ | ✅ | 单行, 大矩阵 | - |
| MaxPool2d | ✅ | ✅ | padding, batch | - |
| Sigmoid | ✅ | ✅ | - | - |
| Tanh | ✅ | ✅ | 极值 | - |
| Dropout | ✅ | ✅ | 训练/推理模式 | - |
| CrossEntropy | ✅ | ✅ | uniform logits | - |
| SGD Update | ✅ | - | 零梯度 | - |
| Flatten | ✅ | ✅ | MNIST size | - |

**总计**: 78 个测试，**100% 通过率**

## 核心测试详解

### Conv2d cuBLAS 测试 (test_conv2d_cublas.cpp) ★核心

```cpp
// 测试 1: ForwardCorrectness
// 验证 cuBLAS forward 与 im2col 结果匹配
// 矩阵: N=64, C=16, H=28, out_C=32
// 误差阈值: 1e-4

// 测试 2: BackwardCorrectness  
// 验证 cuBLAS backward 梯度计算正确
// grad_bias sum: 16.0 (每 channel)
// grad_weight sum: 620.0 (累加)

// 测试 3: CompareWithIm2col
// 验证 cuBLAS vs im2col forward 匹配
// 误差阈值: 1e-4
```

### ReLU 测试 (test_relu.cpp) ★修复验证

```cpp
// 测试: Out-of-place correctness
// 验证 input 保留，output 为 post-ReLU
// 确保 backward mask 正确

// 测试: Backward with negative values
// 验证 negative 区域 gradient 为 0
// 正确使用 pre-ReLU 值作为 mask
```

### Winograd F(6×6) 测试 (test_conv2d_winograd_f6.cpp)

```cpp
// 测试 1: BasicCorrectness
// 输入: 8×8 全1, 权重: 3×3 全1, pad=1
// 输出: 6×6 全9, 与 im2col 匹配

// 测试 2: MultiChannel
// 输入: C=2, out_C=2, 验证多通道累加

// 测试 3: LargerInput
// 输入: 14×14 → 12×12 输出 (2×2 tiles)

// 测试 4: NonUniformInput
// 输入: 递增序列, 验证变换矩阵正确性
```

## 误差标准

| 算子 | 误差阈值 |
|------|---------|
| ReLU | 0 (精确) |
| BiasAdd | 1e-6 |
| Softmax | 1e-5 |
| MatMul FP32 | 5e-2 (大矩阵累积误差) |
| MatMul cuBLAS | 1e-3 |
| Conv2d | 1e-4 |
| Conv2d cuBLAS | 1e-4 ★ |
| Winograd | 1e-2 ~ 2.0 (变换精度) |

## 测试方法

每个测试包含：
1. **CPU 参考实现** - 用 C++/NumPy 计算期望结果
2. **CUDA 实现** - 调用算子 kernel
3. **误差比较** - `relative_error < threshold`

```cpp
float relative_error = fabs(cuda_result - cpu_result) / (fabs(cpu_result) + 1e-6);
EXPECT_LT(relative_error, threshold);
```

## 最近修复

| 日期 | Bug | 修复 |
|------|-----|------|
| 2026-04-30 | ReLU backward mask 问题 | Out-of-place kernel，保存 pre-ReLU 值 ★ |
| 2026-04-30 | 梯度爆炸导致训练失败 | 梯度裁剪 (max_norm=10.0) ★ |
| 2026-04-30 | Conv2d cuBLAS backward 正确性 | reshape + sgemm + col2im |
| 2026-04-30 | Winograd F(6×6) 输出不正确 | wincnn 变换矩阵 |
| 2026-04-30 | cuBLAS 参数错误 | Row-major vs Column-major 处理 |
| 2026-04-29 | WSL2 CUDA 检测失败 | LD_PRELOAD nvidia/cuda 库 |
| 2026-04-29 | ReLU NaN 输入被转为 0 | 条件判断保留 NaN |

## Python 训练测试

```bash
# 测试模型正确性
python3 python/model_cnn_cublas.py

# 训练 benchmark
python3 python/benchmark_cnn_comparison.py

# 输出示例
# CUDA: 6469 samples/s, 88.35% accuracy
# PyTorch: 8372 samples/s, 97.74% accuracy
# Speed ratio: 77%
```

## 测试文件索引

| 文件 | 测试数 | 内容 |
|------|--------|------|
| test_conv2d_cublas.cpp | 3 | Conv2d cuBLAS forward/backward ★核心 |
| test_matmul.cpp | 5 | FP32 Tiled MatMul |
| test_matmul_cublas.cpp | 2 | cuBLAS MatMul |
| test_relu.cpp | 5 | ReLU + out-of-place ★ |
| test_softmax.cpp | 4 | Warp-level Softmax |
| test_conv2d.cpp | 6 | Naive + im2col |
| test_conv2d_winograd.cpp | 2 | Winograd F(2×2) |
| test_conv2d_winograd_f6.cpp | 4 | Winograd F(6×6) |
| test_conv2d_fused.cpp | 1 | Kernel Fusion |
| test_fp16_tensor_core.cpp | 4 | FP16/Tensor Core |
| test_maxpool2d.cpp | 3 | MaxPool2d |
| test_cross_entropy.cpp | 3 | CrossEntropy |
| test_edge_cases.cpp | 9 | 边界场景 |

---

*文档版本: 2.0.0 - 2026-04-30*