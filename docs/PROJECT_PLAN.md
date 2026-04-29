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

### ✅ 初级目标 (已完成)
- 实现矩阵乘法、ReLU、BiasAdd、Softmax
- 正确性验证：与 PyTorch 结果对比（误差 < 1e-6）
- 性能测试：MatMul 1062 GFLOPS (13% T4 峰值)

### ✅ 中级目标 (已完成)
- 实现 Conv2d（支持 NCHW 格式） - 921 GFLOPS
- 实现 MaxPool2d
- 支持反向传播 - 所有算子均有 backward

### ✅ 高级目标 (已完成)
- 实现 Sigmoid、Tanh、Dropout、CrossEntropy、Flatten
- 性能优化：Shared Memory Tiling、Warp-Level Reduction、Vectorized Access、im2col + GEMM
- 端到端：MNIST MLP 训练达到 95.36% 准确率

### ✅ CNN 训练目标 (已完成)
- Conv2d/MaxPool2d Python binding 实现
- SimpleCNN_CUDA 模型：2-Conv CNN 完整训练流程
- MNIST CNN 训练达到 **97.92%** 准确率
- 性能优化：im2col + GEMM (+4.7x)、预分配缓冲区 (+27%)
- PyTorch 对比：准确率相当，速度差距42x（教育性实现预期）

### 🎯 项目状态
- **12 个算子**，forward + backward 全部实现
- **Conv2d/MaxPool2d Python binding**，支持 CNN 训练
- **SimpleCNN_CUDA 模型**，完整 2-Conv CNN 训练流程
- **59 个测试**，100% 通过率（含9个边界情况测试）
- **MLP: 10.61x 性能提升** vs NumPy 实现
- **CNN: 97.92% 准确率**，190→1000 samples/s (+5.3x)

## 5. 产出物

```
.
├── CMakeLists.txt
├── include/
│   ├── cuda_ops.h          # 算子声明 + Conv2dBackwardBuffers
│   └── cuda_util.h         # 工具函数
├── src/
│   ├── matmul.cu           # Tiled GEMM + transpose kernels
│   ├── bias_add.cu         # Broadcasting
│   ├── relu.cu             # Vectorized ReLU
│   ├── softmax.cu          # Warp-level Softmax
│   ├── conv2d.cu           # im2col + GEMM + optimized backward
│   ├── maxpool2d.cu        # Pooling forward/backward
│   ├── cross_entropy.cu    # Loss function
│   ├── sgd_update.cu       # Optimizer
│   ├── flatten.cu          # Reshape
│   ├── sigmoid.cu          # Sigmoid
│   ├── tanh.cu             # Tanh
│   ├── dropout.cu          # Dropout
│   └── cuda_ops_export.cu  # Python C API
├── python/
│   ├── cuda_ops.py         # ctypes binding + 预分配API
│   ├── model_cuda.py       # 纯 CUDA MLP
│   ├── model_cnn_cuda.py   # 纯 CUDA CNN (预分配优化)
│   ├── train_mnist_cuda.py
│   ├── train_mnist_cnn_cuda.py
│   ├── train_mnist_cnn_pytorch.py  # PyTorch对比
│   └── mnist_data.py
├── tests/
│   ├── test_matmul.cpp     # 5 tests
│   ├── test_relu.cpp       # 5 tests
│   ├── test_softmax.cpp    # 4 tests
│   ├── test_conv2d.cpp     # 6 tests
│   ├── test_edge_cases.cpp # 9 tests
│   └ ...                   # 共 59 tests
├── docs/
│   ├── PROJECT_PLAN.md     # 项目计划
│   ├── PERFORMANCE_METRICS.md  # 性能报告 + CNN对比
│   └ README.md
└── build/
    └── lib/libcuda_ops_shared.so  # Python binding共享库
```

每个算子需包含：
1. Forward kernel 实现
2. Backward kernel 实现（支持训练）
3. 正确性测试（GoogleTest + PyTorch对比）
4. 性能 Benchmark

## 6. 项目结构设计原则

- **零依赖**: 除 CUDA/PyTorch 外不引入额外库
- **单算子单文件**: 便于学习和维护
- **测试驱动**: 先写测试再写实现
- **可验证**: 每个算子都必须通过 PyTorch 对比验证

## 7. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch ATen source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda)
- [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/)