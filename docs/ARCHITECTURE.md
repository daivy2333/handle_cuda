# Architecture

## 项目结构

```
handle_cuda/
├── src/                   # CUDA kernels (.cu)
│   ├── matmul.cu          # FP32 Tiled GEMM (1062 GFLOPS)
│   ├── matmul_cublas.cu   # cuBLAS sgemm backend (7869 GFLOPS) ★NEW
│   ├── matmul_fp16.cu     # FP16/Tensor Core MatMul + Backward ★优化
│   ├── half_utils.cu      # FP32/FP16 转换工具
│   ├── loss_scaling.cu    # 梯度缩放 (混合精度训练) ★NEW
│   ├── relu.cu            # Vectorized ReLU
│   ├── softmax.cu         # Warp-level Softmax
│   ├── bias_add.cu        # Broadcasting
│   ├── conv2d.cu          # Naive + im2col + GEMM
│   ├── conv2d_winograd.cu # Winograd F(2×2, 3×3)
│   ├── conv2d_winograd_f6.cu # Winograd F(6×6, 3×3) ★NEW
│   ├── conv2d_fused.cu    # Conv→ReLU Kernel Fusion
│   ├── conv2d_simple.cu   # Simple im2col reference
│   ├── maxpool2d.cu       # Pooling
│   ├── sigmoid.cu         # Sigmoid activation
│   ├── tanh.cu            # Tanh activation
│   ├── dropout.cu         # Dropout
│   ├── cross_entropy.cu   # Cross entropy loss
│   ├── sgd_update.cu      # SGD optimizer
│   ├── flatten.cu         # Reshape
│   └── cuda_ops_export.cu # C API export
│
├── include/
│   ├── cuda_ops.h         # Public API
│   └── cuda_util.h        # Internal utilities + CUBLAS_CHECK
│
├── python/
│   ├── cuda_ops.py        # ctypes binding (FP16 + 梯度缩放)
│   ├── model_cuda.py      # Pure CUDA CNN + Mixed Precision MLP
│   ├── model.py           # NumPy MLP (reference)
│   ├── train_mnist_cuda.py# Training script
│   ├── benchmark_compare.py# PyTorch vs CUDA comparison
│   └── mnist_data.py      # Data loader
│
├── tests/                 # GoogleTest (78 tests, 97% pass)
│   ├── test_matmul_cublas.cpp   # cuBLAS 测试
│   ├── test_conv2d_winograd_f6.cpp # Winograd F6 测试
│   ├── test_fp16_mixed_precision.cpp # FP16 混合精度测试 ★NEW
│   ├── test_tensor_core_optimized.cpp # Tensor Core 优化测试 ★NEW
│   ├── test_edge_cases.cpp # Boundary tests (9 tests)
│   ├── test_conv2d_winograd.cpp # Winograd F(2×2) 测试
│   ├── test_fp16_tensor_core.cpp # FP16/Tensor Core tests
│   ├── test_conv2d_fused.cpp # Kernel Fusion tests
│   └── ...                 # Operator tests
│
├── scripts/
│   └── run_with_cuda.sh   # WSL2 CUDA 环境脚本 ★NEW
│
└── docs/
    ├── PERFORMANCE_METRICS.md # 性能报告 (v1.6.0)
    ├── ARCHITECTURE.md        # 本文档
    ├── TESTING.md             # 测试说明
    └── CUDA_GUIDE.md          # CUDA 指南
```

## 已实现算子

| 算子 | Forward | Backward | 优化技术 | 性能 |
|------|---------|----------|----------|------|
| **MatMul FP32** | ✅ | ✅ | 32×32 Shared Memory Tiling | 1062 GFLOPS |
| **MatMul cuBLAS** | ✅ | ✅ | cuBLAS sgemm backend | **7869 GFLOPS** |
| **MatMul FP16** | ✅ | ✅ | Tensor Core WMMA | 1211 GFLOPS (1.07x) |
| **MatMul FP16 Backward** | ✅ | ✅ | Tiled Kernel (精度修复) | max_error=0.0009 ★修复 |
| **MatMul Tensor Core Opt** | ⚠️ 实验性 | - | Multi-warp + Shared staging | 1.06x (布局问题) |
| **ReLU** | ✅ | ✅ | float4 Vectorization | 200 GB/s |
| **Softmax** | ✅ | ✅ | Warp-Level Reduction | 249 GB/s |
| **BiasAdd** | ✅ | ✅ | Broadcasting | - |
| **Conv2d Naive** | ✅ | ✅ | - | 2.58 GFLOPS |
| **Conv2d im2col** | ✅ | ✅ | im2col + Tiled GEMM | 921 GFLOPS |
| **Conv2d Winograd F(2×2)** | ✅ | - | 4×4 tile | 理论 2.25x |
| **Conv2d Winograd F(6×6)** | ✅ | - | 8×8 tile | **理论 5x** |
| **Conv+ReLU Fused** | ✅ | ✅ | Kernel Fusion | ~1.2x |
| **MaxPool2d** | ✅ | ✅ | - | - |
| **Sigmoid** | ✅ | ✅ | - | - |
| **Tanh** | ✅ | ✅ | - | - |
| **Dropout** | ✅ | ✅ | - | - |
| **CrossEntropy** | ✅ | ✅ | Numerical stability | - |
| **SGD Update** | ✅ | - | - | - |
| **Flatten** | ✅ | ✅ | Memory copy | - |

## 优化技术详解

### 1. MatMul cuBLAS: 行业标准后端

```
使用 NVIDIA cuBLAS 库的 sgemm 函数

┌─────────────────────────────────────┐
│  cuBLAS sgemm 调度                   │
│  ┌─────────────────────────────────┐│
│  │ 深度优化的 Tensor Core 调度      ││
│  │ 多级流水线                       ││
│  │ 自适应 block size 选择           ││
│  └─────────────────────────────────┘│
│                                      │
│  RTX 4060 FP32 峰值: ~13 TFLOPS     │
│  实测性能: 7869 GFLOPS (60% 峰值)   │
└─────────────────────────────────────┘

效果: 7.4x vs 自实现 Tiled MatMul
性能: 7869 GFLOPS @ 2048×2048
适用: 所有矩阵乘法场景（推荐默认使用）
```

### 2. MatMul FP32: Shared Memory Tiling

```
每个 thread block 处理 32×32 输出 tile

┌─────────────────────────────────────┐
│  A tile (32×32)    B tile (32×32)    │
│  ┌──────────────┐  ┌──────────────┐  │
│  │ Shared Mem   │  │ Shared Mem   │  │
│  │ 32×32 floats │  │ 32×32 floats │  │
│  └──────────────┘  └──────────────┘  │
│                                      │
│  每个 thread 计算 C[row][col]        │
│  遍历 K/32 个 tiles                  │
└─────────────────────────────────────┘

效果: 全局内存访问减少 32 倍
性能: 1062 GFLOPS (2048×2048, FP32)
```

### 3. MatMul FP16: Tensor Core (WMMA)

```cpp
// 使用 WMMA API 进行 Tensor Core 计算
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
fragment<accumulator, 16, 16, 16, half> c_frag;

load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, 16, mem_row_major);

效果: 利用 Tensor Core 硬件加速
性能: 1211 GFLOPS (2048×2048, FP16)
加速: 1.07x vs FP32 Tiled
```

### 4. Winograd F(6×6, 3×3) ★NEW

```
Winograd F(6×6) - cuDNN 同等 tile

输入变换:  V = B^T @ U @ B (8×8)
权重变换:  W = G @ w @ G^T (8×8)
元素乘法:  M = V ⊙ W (element-wise, 8×8)
输出变换:  Y = A^T @ M @ A (6×6 输出)

变换矩阵 (wincnn.cookToomFilter([0,1,2,3,4,5,-1], 6, 3)):
A^T = Vandermonde 矩阵 (值 1, 2, 4, 8, 16, 32 powers)
G   = 权重变换矩阵
B^T = 输入变换矩阵 (大值，最高 3125)

输入 tile: 8×8 (m+r-1 = 6+3-1 = 8)
输出 tile: 6×6
效果: 81× → 36× 乘法 (每 tile，实际 2.25x)
优势: 相比 F(2×2)，更大并行度 (36 输出/tile vs 4 输出/tile)
适用: stride=1, pad=1, 3×3 kernel
```

### 5. Winograd F(2×2, 3×3)

```
标准 Winograd 变换 (wincnn matrices):

输入变换:  V = B^T @ U @ B
权重变换:  W = G @ w @ G^T
元素乘法:  M = V ⊙ W (element-wise)
输出变换:  Y = A^T @ M @ A

变换矩阵:
A^T = [[1, 1, 1, 0], [0, 1, -1, 1]]
G   = [[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]]
B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]

效果: 3×3 卷积乘法从 9 次 → 4 次 (每输出像素)
理论: 2.25x 减少乘法
适用: stride=1, pad=1, 3×3 kernel
```

### 6. Kernel Fusion: Conv→ReLU

```cpp
// 融合 kernel: Conv2d + ReLU 在单 kernel 中完成
__global__ void fused_conv_relu_kernel(
    const float* input, const float* weight, float* output, ...) {
    
    // 1. 计算卷积输出 (在 shared memory 或 register)
    float conv_out = compute_conv_pixel(...);
    
    // 2. 直接应用 ReLU，避免写入全局内存再读取
    output[idx] = conv_out > 0 ? conv_out : 0;
}

效果: 
- 减少一次 kernel launch (~5-10μs)
- 减少一次全局内存读写
- Conv 输出直接在 register/shared memory 中处理

性能: ~1.2x 加速 vs 分离 kernels
```

### 7. Softmax: Warp-Level Reduction

```cpp
// 使用 shuffle 指令在 warp 内做 reduction
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 每个 warp (32 threads) 处理一个 batch
效果: 串行 max/sum → 并行 warp reduction
性能: 249 GB/s, 17.8x vs naive
```

### 8. Conv2d im2col: 矩阵化方法

```
input [N, C, H, W] → im2col → col [C×K², N×out_H×out_W]
                               ↓
                     MatMul (复用优化后的 kernel)
                               ↓
                     output [N, out_C, out_H, out_W]

效果: 复用优化后的 MatMul kernel
性能: 921 GFLOPS, 357x vs naive
```

### 9. ReLU: Vectorized Memory Access

```cpp
// 使用 float4 一次读写 4 个元素
float4 data = *reinterpret_cast<float4*>(ptr);

效果: 更好的内存带宽利用
性能: 200 GB/s, 4x vs naive
```

## Kernel 实现细节

### cuBLAS Backend Kernel

```cpp
// matmul_cublas.cu 实现:

// 静态 handle 初始化 (延迟初始化，线程安全)
static cublasHandle_t handle = []{
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    return h;
}();

// cuBLAS sgemm 调用
// 注意: cuBLAS 使用列优先存储，需要调整维度顺序
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,              // 交换 M, N 顺序
    &alpha,
    B, N,                 // B 是 M×K 在列优先下是 K×N
    A, K,                 // A 是 M×K 在列优先下是 M×K
    &beta,
    C, N);
```

### Winograd F(6×6) Kernel 结构 ★NEW

```cpp
// conv2d_winograd_f6.cu 包含 4 个 kernels:

// 1. 权重变换 (预计算，一次)
winograd_f6_weight_transform_kernel
  输入: weight [out_C, C, 3, 3]
  输出: W [out_C, C, 8, 8] = G @ w @ G^T

// 2. 输入变换 (每 batch)
winograd_f6_input_transform_kernel
  输入: input tile U [8, 8] (含 padding)
  输出: V [8, 8] = B^T @ U @ B

// 3. 元素乘法
winograd_f6_elementwise_kernel
  输入: V, W
  输出: M = V ⊙ W (逐元素，累加输入通道)

// 4. 输出变换
winograd_f6_output_transform_kernel
  输入: M [8, 8]
  输出: Y [6, 6] = A^T @ M @ A
```

### Winograd F(2×2) Kernel 结构

```cpp
// conv2d_winograd.cu 包含 4 个 kernels:

// 1. 权重变换 (预计算，一次)
winograd_weight_transform_kernel
  输入: weight [out_C, C, 3, 3]
  输出: W [out_C, C, 4, 4] = G @ w @ G^T

// 2. 输入变换 (每 batch)
winograd_input_transform_kernel
  输入: input tile U [4, 4]
  输出: V [4, 4] = B^T @ U @ B

// 3. 元素乘法
winograd_elementwise_kernel
  输入: V, W
  输出: M = V ⊙ W (逐元素)

// 4. 输出变换
winograd_output_transform_kernel
  输入: M [4, 4]
  输出: Y [2, 2] = A^T @ M @ A
```

### FP16 Tensor Core Kernel 结构

```cpp
// matmul_fp16.cu 实现:

// 1. FP32 → FP16 转换
cuda_fp32_to_fp16(input_fp32, input_fp16, size);

// 2. WMMA Tensor Core 计算
wmma_matmul_kernel<<<...>>>(A_fp16, B_fp16, C_fp16, M, N, K);

// 3. FP16 → FP32 转换 (可选)
cuda_fp16_to_fp32(output_fp16, output_fp32, size);
```

## 设计原则

1. **模块化设计** - 每个优化变体单独文件，便于选择
2. **零外部依赖** - 仅依赖 CUDA Toolkit 和 GoogleTest (cuBLAS 为 CUDA 内置)
3. **测试驱动** - 72 个单元测试，覆盖 forward/backward + 边界场景
4. **可验证** - 与 PyTorch 数值对比，误差 < 1e-6
5. **边界覆盖** - NaN/Inf、显存压力、batch=1、非方阵矩阵
6. **向后传播** - 所有训练算子均有 backward 实现

## API 设计

### C API (cuda_ops_export.cu)

```c
// MatMul 变体
void cuda_matmul_f32(const float* A, const float* B, float* C, 
                      size_t M, size_t N, size_t K);
void cuda_matmul_cublas(const float* A, const float* B, float* C,
                         size_t M, size_t N, size_t K, cudaStream_t stream);
void cuda_matmul_fp16(const half* A, const half* B, half* C,
                       size_t M, size_t N, size_t K);

// Conv2d 变体
void cuda_conv2d_f32(...);            // Naive
void cuda_conv2d_im2col_f32(...);     // im2col + GEMM
void cuda_conv2d_winograd_f32(...);   // Winograd F(2×2)
void cuda_conv2d_winograd_f6(...);    // Winograd F(6×6) ★NEW
void cuda_conv2d_fused_f32(...);      // Conv→ReLU 融合

// 激活函数
void cuda_relu_f32(float* data, size_t size);
void cuda_softmax_f32(const float* input, float* output,
                      size_t batch, size_t classes);
```

### Python Binding (cuda_ops.py)

```python
class CUDAOps:
    def matmul(self, A, B, M, N, K): ...
    def matmul_cublas(self, A, B, M, N, K): ...      # ★NEW
    def matmul_fp16(self, A, B, M, N, K): ...
    def conv2d(self, ...): ...              # 自动选择 im2col
    def conv2d_im2col(self, ...): ...       # im2col + GEMM
    def conv2d_winograd(self, ...): ...     # Winograd F(2×2)
    def conv2d_winograd_f6(self, ...): ...  # Winograd F(6×6) ★NEW
    def relu(self, data, size): ...
    def softmax(self, input, output, batch, classes): ...
```

## 性能选择指南

| 场景 | 推荐实现 | 原因 |
|------|----------|------|
| **大矩阵乘法** | MatMul cuBLAS | 7869 GFLOPS，接近峰值 |
| 小矩阵 (<512) | MatMul FP32 | cuBLAS 启动开销 |
| FP16 训练 | MatMul FP16 | Tensor Core 加速 |
| **3×3 卷积 stride=1** | Winograd F(6×6) | 与 cuDNN 同等 tile |
| 其他卷积 | im2col + cuBLAS | 通用性 |
| Conv→ReLU | Fused Kernel | 减少内存读写 |
| 边界检查场景 | Naive Conv2d | 调试/验证 |

## WSL2 环境支持

```bash
# WSL2 CUDA 检测问题解决方案
# scripts/run_with_cuda.sh
export LD_PRELOAD=/usr/lib/wsl/lib/libnvidia-ml.so.1:/usr/lib/wsl/lib/libcuda.so.1

# 使用方式
source scripts/run_with_cuda.sh
ctest --output-on-failure

# 或直接
./bin/test_matmul_cublas
```

---

*文档版本: 1.4.0 - 2026-04-30*