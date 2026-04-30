# Testing Guide

## 测试框架

使用 GoogleTest 作为单元测试框架。

## 运行测试

```bash
# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行所有测试 (66 tests)
ctest --output-on-failure

# 运行单个测试
./bin/test_matmul
./bin/test_relu
./bin/test_softmax
./bin/test_conv2d
./bin/test_conv2d_winograd    # Winograd 测试
./bin/test_fp16_tensor_core   # Tensor Core 测试
./bin/test_conv2d_fused       # Kernel Fusion 测试
./bin/test_edge_cases         # 边界测试
```

## 测试覆盖

| 算子 | Forward | Backward | 边界测试 | 性能测试 |
|------|---------|----------|----------|----------|
| MatMul FP32 | ✅ | ✅ | 转置, 大矩阵, 非方阵 | ✅ |
| MatMul FP16 | ✅ | ✅ | FP32/FP16 转换 | ✅ |
| ReLU | ✅ | ✅ | 全正, 全负, NaN/Inf | ✅ |
| Softmax | ✅ | ✅ | 和为1, 非负, batch=1, NaN/Inf | ✅ |
| BiasAdd | ✅ | ✅ | 单行, 大矩阵 | - |
| Conv2d Naive | ✅ | ✅ | 无padding, 非方阵 | - |
| Conv2d im2col | ✅ | ✅ | im2col 验证 | ✅ |
| Conv2d Winograd | ✅ | - | im2col 对比 | ✅ |
| Conv+ReLU Fused | ✅ | ✅ | 分离 kernel 对比 | ✅ |
| MaxPool2d | ✅ | ✅ | padding, batch | - |
| Sigmoid | ✅ | ✅ | - | - |
| Tanh | ✅ | ✅ | 极值 | - |
| Dropout | ✅ | ✅ | 训练/推理模式 | - |
| CrossEntropy | ✅ | ✅ | uniform logits | - |
| SGD Update | ✅ | - | 零梯度 | - |
| Flatten | ✅ | ✅ | MNIST size | - |

**边界测试 (Edge Cases)**:
- 非方阵矩阵：MatMul 512×2048, Conv2d 非对称输入
- batch_size 边界：batch=1（最小批次）
- 显存压力：4096×4096, 8192×8192（显存不足时自动跳过）
- NaN/Inf 容错：Softmax +Inf 输入, ReLU NaN 输入

**总计**: 66 个测试，**100% 通过率**

## 新增测试详解

### Winograd 测试 (test_conv2d_winograd.cpp)

```cpp
// 测试 1: DebugTileMapping
// 验证 Winograd tile 映射正确性
// 输入: 4×4 全1, 权重: 3×3 全1
// 期望输出: 与 im2col 匹配 (4, 6, 6, 9)

// 测试 2: SimpleConvReference  
// 验证 im2col 在更大输入上的正确性
// 输入: 5×5 序列, 权重: 3×3 全1
// 期望: 直接卷积 vs im2col 匹配
```

### FP16 Tensor Core 测试 (test_fp16_tensor_core.cpp)

```cpp
// 测试 1: FP16Conversion
// 验证 FP32 ↔ FP16 转换正确性

// 测试 2: MatmulCorrectness
// 验证 FP16 matmul 结果与 FP32 匹配 (误差 < 1e-3)

// 测试 3: MatmulPerformance
// 测量 FP16 vs FP32 加速比

// 测试 4: SmallMatmul
// 验证小矩阵 Tensor Core 正确性
```

### Kernel Fusion 测试 (test_conv2d_fused.cpp)

```cpp
// 测试: ConvReluCorrectness
// 验证融合 kernel 与分离 kernels 结果匹配
// Conv2d im2col → ReLU == Conv2d+ReLU fused
```

## 误差标准

| 算子 | 误差阈值 |
|------|---------|
| ReLU | 0 (精确) |
| BiasAdd | 1e-6 |
| Softmax | 1e-5 |
| MatMul FP32 | 5e-2 (大矩阵浮点累积误差) |
| MatMul FP16 | 1e-3 (FP16 精度限制) |
| Conv2d | 1e-4 |
| Winograd | 1e-2 (变换矩阵误差) |
| Kernel Fusion | 0 (精确) |

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
| 2026-04-30 | Winograd 输出不匹配 im2col | 修正 A, G, B 变换矩阵 (wincnn 标准) |
| 2026-04-30 | Winograd 输入变换公式错误 | B^T @ U @ B (修正 B 矩阵使用) |
| 2026-04-30 | Winograd 输出变换索引错误 | A[:,p]^T @ M @ A[:,q] (p,q 0-based) |
| 2026-04-29 | ReLU NaN 输入被转为 0 | 改用条件判断保留 NaN |
| 2026-04-29 | Softmax +Inf 输出 NaN | 特殊处理 +Inf 情况 |
| 2026-04-29 | MaxPool2d blockDim 问题 | 改为 (1,1)，修复 out_H/out_W 计算 |
| 2026-04-29 | 测试期望值错误 | 用 PyTorch 验证修正 |

## 性能测试输出示例

```
========== Tensor Core Matmul Performance ==========
  FP32: 1128.03 GFLOPS (1.90375 ms)
  FP16+TensorCore: 1211.29 GFLOPS (1.77289 ms)
  Speedup: 1.07381x
=====================================================

Winograd output: 4.00 6.00 6.00 9.00 
im2col output:   4.00 6.00 6.00 9.00 
[PASS] Winograd matches im2col
```

---

*文档版本: 1.3.0 - 2026-04-30*