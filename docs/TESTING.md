# Testing Guide

## 测试框架

使用 GoogleTest 作为单元测试框架。

## 运行测试

```bash
# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行所有测试 (50 tests)
ctest --output-on-failure

# 运行单个测试
./bin/test_matmul
./bin/test_relu
./bin/test_softmax
```

## 测试覆盖

| 算子 | Forward | Backward | 边界测试 | 性能测试 |
|------|---------|----------|----------|----------|
| MatMul | ✅ | ✅ | 转置, 大矩阵 | ✅ |
| ReLU | ✅ | ✅ | 全正, 全负 | ✅ |
| Softmax | ✅ | ✅ | 和为1, 非负 | ✅ |
| BiasAdd | ✅ | ✅ | 单行, 大矩阵 | - |
| Conv2d | ✅ | ✅ | 无padding, im2col | ✅ |
| MaxPool2d | ✅ | ✅ | padding, batch | - |
| Sigmoid | ✅ | ✅ | - | - |
| Tanh | ✅ | ✅ | 极值 | - |
| Dropout | ✅ | ✅ | 训练/推理模式 | - |
| CrossEntropy | ✅ | ✅ | uniform logits | - |
| SGD Update | ✅ | - | 零梯度 | - |
| Flatten | ✅ | ✅ | MNIST size | - |

**总计**: 50 个测试，94% 通过率

## 误差标准

| 算子 | 误差阈值 |
|------|---------|
| ReLU | 0 (精确) |
| BiasAdd | 1e-6 |
| Softmax | 1e-5 |
| MatMul | 1e-2 (大矩阵) |
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

## 已知问题

- `MatMulTest.LargeMatrix` - 大矩阵精度问题 (2.3% vs 1% 预期)
- `MaxPool2dTest.Basic/WithPadding` - 功能 bug (待修复)