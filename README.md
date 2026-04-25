# CUDA Deep Learning Operators

用 CUDA 实现深度学习核心算子，作为 CUDA 编程的练手项目。

## 项目结构

```
.
├── CMakeLists.txt
├── include/
│   ├── cuda_ops.h          # 算子声明
│   └── cuda_util.h         # 工具函数
├── src/
│   ├── matmul.cu           # 矩阵乘法
│   ├── relu.cu             # ReLU 激活
│   ├── bias_add.cu         # 偏置加法
│   ├── softmax.cu          # Softmax
│   ├── conv2d.cu           # 2D 卷积
│   └── maxpool2d.cu        # 2D 最大池化
├── tests/
│   ├── CMakeLists.txt
│   ├── test_matmul.cpp
│   ├── test_relu.cpp
│   ├── test_bias_add.cpp
│   ├── test_softmax.cpp
│   ├── test_conv2d.cpp
│   └── test_maxpool2d.cpp
└── scripts/
    ├── benchmark.py        # PyTorch 性能对比
    └── CMakeLists.txt
```

## 已实现算子

| 算子 | 文件 | 状态 |
|------|------|------|
| MatMul | src/matmul.cu | ✅ |
| BiasAdd | src/bias_add.cu | ✅ |
| ReLU | src/relu.cu | ✅ |
| Softmax | src/softmax.cu | ✅ |
| Conv2d | src/conv2d.cu | ✅ |
| MaxPool2d | src/maxpool2d.cu | ✅ |

## 依赖

- CUDA Toolkit >= 11.0
- CMake >= 3.18
- GoogleTest (自动下载)
- Python3 (用于 benchmark)
- PyTorch (用于 benchmark)

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 测试

```bash
cd build
make run_tests
./bin/test_matmul
./bin/test_relu
./bin/test_bias_add
./bin/test_softmax
./bin/test_conv2d
./bin/test_maxpool2d
```

## 性能对比

```bash
python3 ../scripts/benchmark.py
```

## 实现难度

| 算子 | 难度 | 说明 |
|------|------|------|
| ReLU | ⭐ | Element-wise 操作 |
| BiasAdd | ⭐ | Element-wise + broadcast |
| Softmax | ⭐⭐ | 需要归一化 |
| MatMul | ⭐⭐ | 经典 GEMM |
| MaxPool2d | ⭐⭐ | 索引追踪 |
| Conv2d | ⭐⭐⭐ | Im2Col + GEMM |

## 设计原则

1. **零外部依赖**: 除 CUDA/PyTorch 外不引入额外库
2. **单算子单文件**: 便于学习和维护
3. **测试驱动**: 每个算子都有对应的 GoogleTest 测试
4. **可验证**: 测试中包含与 PyTorch 结果的对比

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch ATen native/cuda](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda)
