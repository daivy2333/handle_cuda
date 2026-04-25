# CUDA Deep Learning Operators

## 1. 项目概述

用CUDA实现深度学习核心算子，作为CUDA编程的练手项目。

## 2. 技术栈

| 类别 | 技术 |
|------|------|
| 核心语言 | CUDA C/C++ (C++17) |
| 构建工具 | CMake |
| 测试框架 | GoogleTest |
| 验证框架 | PyTorch (用于结果对比) |

## 3. 算子实现清单

### Phase 1: MLP 基础算子（最小可运行）

| 算子 | 说明 | 难度 |
|------|------|------|
| `cuda_relu` | ReLU 激活函数 | ⭐ |
| `cuda_bias_add` | 偏置加法 | ⭐ |
| `cuda_matmul` | 矩阵乘法 (FWD) | ⭐⭐ |
| `cuda_softmax` | Softmax 输出层 | ⭐⭐ |

**组合示例**: `MatMul → BiasAdd → ReLU → MatMul → Softmax`

### Phase 2: CNN 核心算子

| 算子 | 说明 | 难度 |
|------|------|------|
| `cuda_conv2d` | 2D 卷积 | ⭐⭐⭐ |
| `cuda_maxpool2d` | 2x2 最大池化 | ⭐⭐ |
| `cuda_relu_gradient` | ReLU 反向传播 | ⭐⭐ |

### Phase 3: 进阶算子

| 算子 | 说明 | 难度 |
|------|------|------|
| `cuda_batchnorm2d` | Batch Normalization | ⭐⭐⭐ |
| `cuda_dropout` | Dropout 正则化 | ⭐⭐ |

## 4. 达到程度

### 初级目标
- 实现矩阵乘法、ReLU、BiasAdd、Softmax
- 正确性验证：与 PyTorch 结果对比（误差 < 1e-6）
- 性能测试：与 cuBLAS 对比（允许 20% 差异）

### 中级目标
- 实现 Conv2d（支持 NCHW 格式）
- 实现 MaxPool2d
- 支持反向传播

### 高级目标
- 实现 BatchNorm
- 性能优化：使用 shared memory、tensor core
- 端到端：能用自己写的算子搭一个可训练的 MLP

## 5. 产出物

```
.
├── CMakeLists.txt
├── include/
│   ├── cuda_ops.h          # 算子声明
│   └── cuda_util.h         # 工具函数
├── src/
│   ├── matmul.cu
│   ├── bias_add.cu
│   ├── relu.cu
│   ├── softmax.cu
│   ├── conv2d.cu
│   └── maxpool2d.cu
├── tests/
│   ├── test_matmul.cpp
│   ├── test_relu.cpp
│   ├── test_conv2d.cpp
│   └── ...
├── scripts/
│   ├── run_tests.sh
│   └── benchmark.py        # 性能对比脚本
└── docs/
    └── implementation.md   # 实现细节文档
```

每个算子需包含：
1. Forward kernel 实现
2. 正确性测试（GoogleTest + PyTorch对比）
3. Benchmark 测试

## 6. 项目结构设计原则

- **零依赖**: 除 CUDA/PyTorch 外不引入额外库
- **单算子单文件**: 便于学习和维护
- **测试驱动**: 先写测试再写实现
- **可验证**: 每个算子都必须通过 PyTorch 对比验证

## 7. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch ATen source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda)
- [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/)
