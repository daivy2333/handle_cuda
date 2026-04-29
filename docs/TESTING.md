# Testing Guide

## 测试框架

使用 GoogleTest 作为单元测试框架。

## 运行测试

```bash
# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行所有测试 (59 tests)
ctest --output-on-failure

# 运行单个测试
./bin/test_matmul
./bin/test_relu
./bin/test_softmax
./bin/test_edge_cases  # 边界测试
```

## 测试覆盖

| 算子 | Forward | Backward | 边界测试 | 性能测试 |
|------|---------|----------|----------|----------|
| MatMul | ✅ | ✅ | 转置, 大矩阵, 非方阵 | ✅ |
| ReLU | ✅ | ✅ | 全正, 全负, NaN/Inf | ✅ |
| Softmax | ✅ | ✅ | 和为1, 非负, batch=1, NaN/Inf | ✅ |
| BiasAdd | ✅ | ✅ | 单行, 大矩阵 | - |
| Conv2d | ✅ | ✅ | 无padding, im2col, 非方阵输入 | ✅ |
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

**总计**: 59 个测试，**100% 通过率**

## 误差标准

| 算子 | 误差阈值 |
|------|---------|
| ReLU | 0 (精确) |
| BiasAdd | 1e-6 |
| Softmax | 1e-5 |
| MatMul | 5e-2 (大矩阵浮点累积误差) |
| Conv2d | 1e-4 |

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
| 2026-04-29 | ReLU NaN 输入被转为 0 | 改用条件判断保留 NaN |
| 2026-04-29 | Softmax +Inf 输出 NaN | 特殊处理 +Inf 情况 |
| 2026-04-29 | MaxPool2d blockDim 问题 | 改为 (1,1)，修复 out_H/out_W 计算 |
| 2026-04-29 | 测试期望值错误 | 用 PyTorch 验证修正 |