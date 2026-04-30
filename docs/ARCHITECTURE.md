# Architecture

## 项目概述

本项目实现了一个纯 CUDA 的 CNN 训练框架，用于 MNIST 手写数字识别。目标是复现 PyTorch 的核心训练流程，实现高性能的 GPU 算子。

**最终性能**: CUDA 实现达到 PyTorch **77%** 的训练速度，MNIST 测试准确率 **88.35%**。

## 项目结构

```
handle_cuda/
├── src/                   # CUDA kernels (.cu)
│   ├── matmul.cu          # FP32 Tiled GEMM (1062 GFLOPS)
│   ├── matmul_cublas.cu   # cuBLAS sgemm backend (7869 GFLOPS)
│   ├── matmul_fp16.cu     # FP16/Tensor Core MatMul + Backward
│   ├── conv2d.cu          # Naive + im2col + GEMM
│   ├── conv2d_cublas.cu   # Conv2d cuBLAS backend (训练用) ★核心
│   ├── conv2d_winograd.cu # Winograd F(2×2, 3×3)
│   ├── conv2d_winograd_f6.cu # Winograd F(6×6, 3×3)
│   ├── conv2d_fused.cu    # Conv→ReLU Kernel Fusion
│   ├── relu.cu            # Vectorized ReLU + Out-of-place ★修复
│   ├── softmax.cu         # Warp-level Softmax
│   ├── cross_entropy.cu   # Cross entropy loss
│   ├── sgd_update.cu      # SGD optimizer
│   ├── maxpool2d.cu       # Pooling
│   ├── flatten.cu         # Reshape
│   └── cuda_ops_export.cu # C API export
│
├── include/
│   ├── cuda_ops.h         # Public API
│   └── cuda_util.h        # Internal utilities
│
├── python/
│   ├── cuda_ops.py        # ctypes binding + 梯度裁剪 ★核心
│   ├── model_cnn_cublas.py # Pure CUDA CNN (cuBLAS backend) ★核心
│   ├── model_cnn_cuda.py  # Pure CUDA CNN (im2col backend)
│   ├── model_cuda.py      # Pure CUDA MLP + Mixed Precision
│   ├── benchmark_cnn_comparison.py # CUDA vs PyTorch benchmark
│   └── mnist_data.py      # Data loader (PyTorch normalization)
│
├── tests/                 # GoogleTest (78 tests)
│   ├── test_conv2d_cublas.cpp # Conv2d cuBLAS 测试 ★核心
│   ├── test_matmul_cublas.cpp # cuBLAS MatMul 测试
│   ├── test_fp16_tensor_core.cpp
│   ├── test_conv2d_winograd_f6.cpp
│   └── ...                 # Operator tests
│
├── scripts/
│   └── run_with_cuda.sh   # WSL2 CUDA 环境脚本
│
└── docs/
    ├── PERFORMANCE_METRICS.md # 性能报告
    ├── ARCHITECTURE.md        # 本文档
    └── TESTING.md             # 测试说明
```

## CNN 模型架构

```
SimpleCNN (MNIST):
Input [batch, 1, 28, 28]
  → Conv1(1→16, kernel=3x3, stride=1, padding=1) [batch, 16, 28, 28]
  → ReLU (out-of-place, 保存 pre-ReLU 值用于 backward)
  → MaxPool(2x2, stride=2) [batch, 16, 14, 14]
  → Conv2(16→32, kernel=3x3, stride=1, padding=1) [batch, 32, 14, 14]
  → ReLU (out-of-place)
  → MaxPool(2x2, stride=2) [batch, 32, 7, 7]
  → Flatten [batch, 1568]
  → FC(1568→10) [batch, 10]
  → Softmax + CrossEntropy

训练配置:
- batch_size: 64
- learning_rate: 0.01
- optimizer: SGD with gradient clipping (max_norm=10.0)
- epochs: 5
```

## 已实现算子

| 算子 | Forward | Backward | 优化技术 | 性能 |
|------|---------|----------|----------|------|
| **MatMul FP32** | ✅ | ✅ | 32×32 Tiling | 1062 GFLOPS |
| **MatMul cuBLAS** | ✅ | ✅ | cuBLAS sgemm | **7869 GFLOPS** |
| **MatMul FP16** | ✅ | ✅ | Tensor Core WMMA | 1211 GFLOPS |
| **ReLU** | ✅ | ✅ | float4 + out-of-place | 正确性修复 ★ |
| **Softmax** | ✅ | ✅ | Warp Reduction | 249 GB/s |
| **Conv2d im2col** | ✅ | ✅ | im2col + Tiled GEMM | 921 GFLOPS |
| **Conv2d cuBLAS** | ✅ | ✅ | im2col + cuBLAS sgemm | 训练核心 ★ |
| **Conv2d Winograd** | ✅ | - | F(2×2) / F(6×6) | 理论 2.25x/5x |
| **Conv+ReLU Fused** | ✅ | ✅ | Kernel Fusion | ~1.2x |
| **MaxPool2d** | ✅ | ✅ | Index caching | - |
| **CrossEntropy** | ✅ | ✅ | Softmax fusion | 数值稳定 |
| **SGD Update** | ✅ | - | Gradient clipping ★ | 防止爆炸 |
| **Flatten** | ✅ | ✅ | Memory reshape | - |

## 关键优化技术

### 1. Conv2d cuBLAS Backend (训练核心)

```
Conv2d = im2col + cuBLAS sgemm

┌─────────────────────────────────────┐
│  Forward:                           │
│  1. im2col: input → col_buffer       │
│  2. sgemm: output = weight @ col     │
│  3. reshape + bias add               │
│                                      │
│  Backward:                           │
│  1. reshape grad_output              │
│  2. grad_weight = grad @ col^T       │
│  3. grad_input = weight^T @ grad     │
│  4. col2im: grad_col → grad_input    │
└─────────────────────────────────────┘

性能优势:
- cuBLAS sgemm: 7869 GFLOPS (7.4x vs 自实现)
- 预分配 buffer: 避免 malloc/free 开销
- 完整 backward: 支持端到端训练
```

### 2. Out-of-place ReLU (正确性修复)

```
问题: In-place ReLU 修改了 forward output，
      backward 时无法正确获取 pre-ReLU 值作为 mask

解决: Out-of-place ReLU kernel
      input (pre-ReLU) → output (post-ReLU)
      input 保留用于 backward mask

__global__ void relu_out_of_place_kernel(
    const float* input, float* output, size_t size) {
    float val = input[idx];
    output[idx] = (val > 0.0f) ? val : 0.0f;
}
```

### 3. 梯度裁剪 (防止爆炸)

```
问题: Conv2 backward 梯度偶尔爆炸 (5003.32 → 权重爆炸)

解决: SGD update 前应用梯度裁剪

def update(self, lr, max_grad_norm=10.0):
    # Clip each gradient tensor
    self.ops.gradient_clip(self.g_conv2_w_ptr, size, max_grad_norm)
    
    # Then SGD update
    self.ops.sgd_update(param, grad, size, lr)

效果: 训练稳定，Loss 正常下降
      无裁剪: Epoch 3 准确率降到 63%
      有裁剪: Epoch 5 准确率 88.35%
```

### 4. MatMul cuBLAS Backend

```
性能: 7869 GFLOPS @ 2048×2048×2048

调用方式:
cuda_matmul_cublas(A, B, C, M, N, K, stream);

优势:
- 深度优化的 Tensor Core 调度
- 多级流水线
- 自适应 block size

适用: 所有矩阵乘法场景 (推荐默认)
```

### 5. Winograd Convolution

```
F(6×6, 3×3) - cuDNN 同等 tile size

输入变换: V = B^T @ U @ B (8×8)
权重变换: W = G @ w @ G^T (预计算)
元素乘法: M = V ⊙ W
输出变换: Y = A^T @ M @ A (6×6)

理论加速: 81× → 36× 乘法/tile (实际 ~2x)
适用: stride=1, pad=1, 3×3 kernel
```

## 训练流程

```python
# Forward
logits_ptr = model.forward(x_ptr, batch)

# Forward 步骤:
conv1_out = conv2d_cublas(x, w1, b1)  # cuBLAS sgemm
conv1_relu = relu_out_of_place(conv1_out)  # 保存 pre-ReLU
pool1_out = maxpool2d(conv1_relu)
conv2_out = conv2d_cublas(pool1_out, w2, b2)
conv2_relu = relu_out_of_place(conv2_out)
pool2_out = maxpool2d(conv2_relu)
flat = flatten(pool2_out)
logits = matmul(flat, fc_w) + fc_b

# Backward
loss = model.backward(logits_ptr, targets)

# Backward 步骤:
grad_logits = cross_entropy_loss_backward(logits, targets)
grad_flat, grad_fc_w = matmul_backward(grad_logits, flat, fc_w)
grad_pool2 = flatten_backward(grad_flat)
grad_conv2_relu = maxpool2d_backward(grad_pool2)
grad_conv2 = relu_backward(grad_conv2_relu, conv2_pre_relu)  # 用 pre-ReLU
grad_pool1 = conv2d_cublas_backward(grad_conv2, pool1_out, w2)
grad_conv1_relu = maxpool2d_backward(grad_pool1)
grad_conv1 = relu_backward(grad_conv1_relu, conv1_pre_relu)
grad_x = conv2d_cublas_backward(grad_conv1, x, w1)  # 不需要

# Update (with gradient clipping)
model.update(lr, max_grad_norm=10.0)
```

## 性能对比结果

| 指标 | CUDA cuBLAS | PyTorch |
|------|-------------|---------|
| 训练速度 | 6469 samples/s | 8372 samples/s |
| 相对速度 | **77%** | 100% |
| 最终准确率 | **88.35%** | 97.74% |
| Epoch 时间 | 9.3s | 7.2s |

**速度差距原因**:
- PyTorch 使用 cuDNN 深度优化 kernel
- CUDA 实现的 Python binding 有数据传输开销
- cuBLAS 虽高性能，但 im2col 有额外开销

**准确率差距原因**:
- 梯度裁剪可能过于保守
- ReLU backward 实现可能有精度差异
- PyTorch 有更多隐式优化

## 设计原则

1. **正确性优先** - 所有算子有 backward 实现，梯度裁剪防止爆炸
2. **模块化设计** - 每个优化变体单独文件，便于选择
3. **零外部依赖** - 仅依赖 CUDA Toolkit 和 GoogleTest
4. **测试驱动** - 78 个单元测试，覆盖 forward/backward + 边界场景
5. **数值验证** - 与 PyTorch 对比，误差 < 1e-4

## API 设计

### Python Binding (核心接口)

```python
class CUDAOps:
    # 核心训练接口
    def conv2d_cublas(self, input, weight, bias, N, C, H, W, out_C, kernel, ...): ...
    def conv2d_cublas_backward(self, grad_out, input, weight, ...): ...
    
    # 辅助接口
    def relu_out_of_place(self, input, output, size): ...
    def relu_backward(self, grad_out, pre_relu, grad_in, size): ...
    def maxpool2d(self, input, ...): ...
    def maxpool2d_backward(self, grad_out, indices, ...): ...
    
    # 梯度控制
    def gradient_clip(self, grad_ptr, size, max_norm): ...
    def sgd_update(self, param, grad, size, lr): ...
    
    # 内存管理
    def alloc(self, size): ...
    def to_device(self, arr): ...
    def to_host(self, ptr, shape): ...
    def free(self, ptr): ...
```

## 性能选择指南

| 场景 | 推荐实现 | 原因 |
|------|----------|------|
| **CNN 训练** | Conv2d cuBLAS | 完整 backward，梯度裁剪 |
| 大矩阵乘法 | MatMul cuBLAS | 7869 GFLOPS |
| 3×3 卷积 stride=1 | Winograd F(6×6) | 理论 5x 加速 |
| Conv→ReLU | Fused Kernel | 减少内存读写 |
| 边界检查 | Naive Conv2d | 调试/验证 |

---

*文档版本: 2.0.0 - 2026-04-30*