# CUDA Deep Learning Operators

自己实现的 CUDA 深度学习算子库，支持完整神经网络训练。

## 项目亮点

- **纯 CUDA 实现**：9 个深度学习算子（forward + backward）
- **性能优化**：MatMul 1062 GFLOPS, Softmax 249 GB/s, Conv2d 921 GFLOPS
- **完整训练**：MNIST MLP 训练，准确率 95.36%
- **正确性验证**：与 PyTorch 对比，误差 < 1e-6

## 性能数据

| 算子 | 性能 | 优化技术 | 对比 |
|------|------|----------|------|
| MatMul | 1062 GFLOPS | 32x32 Shared Memory Tiling | ~80% cuBLAS |
| Softmax | 249 GB/s | Warp-Level Reduction (__shfl_down_sync) | 17.8x vs naive |
| ReLU | 200 GB/s | float4 Vectorized Memory Access | 4x vs naive |
| Conv2d | 921 GFLOPS | im2col + Tiled GEMM | 281x vs naive |

## 训练效果

```
MNIST MLP (3-layer):
- Epoch 10: test_acc=95.36%
- Pure CUDA forward/backward: 2.37 ms/batch
- Speedup vs NumPy: 3.84x
```

## 架构

```
CUDA Layer (C++):
├── matmul.cu      - Tiled GEMM kernel
├── relu.cu        - Vectorized activation
├── softmax.cu     - Warp-level reduction
├── bias_add.cu    - Broadcasting
├── conv2d.cu      - im2col + GEMM
├── maxpool2d.cu   - Pooling
├── cross_entropy.cu - Loss function
├── sgd_update.cu  - Optimizer
└── flatten.cu     - Reshape

Python Layer:
├── cuda_ops.py    - ctypes binding
├── model_cuda.py  - Pure CUDA MLP
├── train_mnist_cuda.py - Training loop
└── mnist_data.py  - Data loader
```

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run training
cd python
python train_mnist_cuda.py
```

## 技术细节

### MatMul Optimization
- 32x32 shared memory tiling
- 减少全局内存访问：K -> K/32
- Bank conflict avoidance

### Softmax Optimization
- Warp-level reduction using `__shfl_down_sync`
- 每个 warp (32 threads) 处理一个 batch
- 消除串行 max/sum 计算

### Conv2d Optimization
- im2col 转换 + 优化的 GEMM
- 复用 MatMul tiled kernel
- 内存开销：col_buffer

## 文件结构

```
handle_cuda/
├── src/           # CUDA kernels
├── include/       # Header files
├── python/        # Python binding & training
├── tests/         # GoogleTest unit tests
├── docs/          # Performance report
└── CMakeLists.txt
```

## 参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS/cuDNN Performance](https://docs.nvidia.com/cuda/cublas/)
- [PyTorch ATen Native CUDA](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda)