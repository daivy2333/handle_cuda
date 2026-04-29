# CNN训练性能加速设计文档

## 概述

### 背景

当前纯CUDA CNN训练性能：~1000 samples/s
PyTorch性能：~42,400 samples/s
差距：42x

### 目标

通过手动实现高级优化技术，在保持教育性纯CUDA实现的前提下，尽可能缩小与PyTorch的性能差距。

### 约束

- 尽量不使用官方库（cuBLAS/cuDNN）
- 最大化学习价值
- 无固定性能目标，能力范围内尽可能优化

---

## 优化技术选择

| 技术 | 预期收益 | 实现方式 | 复杂度 |
|------|----------|----------|--------|
| **Winograd算法** | 2.5x (3x3卷积) | 手动F(4×4, 3×3) | 高 |
| **FP16/Tensor Core** | 2-4x | wmma API + half精度 | 高 |
| **Kernel融合** | 1.5-2x | Conv→ReLU→Pool单kernel | 高 |

**预期总体加速**: 5-15x (5000-15000 samples/s)

---

## 模块结构

```
src/
├── conv2d.cu              # 原始实现 (保留作为fallback)
├── conv2d_winograd.cu     # Winograd F(4×4, 3×3) 实现 [新增]
├── conv2d_fused.cu        # Conv→ReLU→Pool 融合kernel [新增]
├── matmul_fp16.cu         # FP16 matmul + Tensor Core [新增]
├── tensor_core.cu         # wmma API封装 [新增]
├── half_utils.h           # FP16工具函数 [新增]
└── cuda_ops.h             # 扩展接口支持新算子

python/
├── cuda_ops.py            # 扩展Python绑定
├── model_cnn_cuda.py      # 可切换使用新算子
└── benchmark_compare.py   # 性能对比脚本 [新增]
```

---

## 算子选择策略

```python
def conv2d_forward(input, weight, ...):
    if kernel_size == 3 and stride == 1 and pad == 1:
        return conv2d_winograd(input, weight)  # Winograd最优
    elif use_tensor_core and input.is_fp16:
        return conv2d_tensor_core(input, weight)  # Tensor Core
    elif fuse_relu_pool:
        return conv2d_fused_relu_pool(input, weight)  # 融合
    else:
        return conv2d_im2col(input, weight)  # fallback
```

---

## Winograd算法设计

### 原理

Winograd F(4×4, 3×3) 将3x3卷积的乘法次数从9次降到4次/输出像素：

```
标准卷积: 每输出像素 = 9次乘法 + 9次加法
Winograd: 每输出像素 = 4次乘法 + 多次加法 (加法便宜)
理论加速: 2.25x
```

### 变换矩阵

F(4×4, 3×3) 的变换矩阵：

```cpp
// B^T: Input transform (4×4)
Bt = [
    [ 1,  0, -1,  0],
    [ 0,  1,  1,  0],
    [ 0, -1,  1,  0],
    [ 0,  1,  0, -1]
]

// G: Weight transform (4×3)
G = [
    [ 1,     0,     0],
    [-2/3, -2/3, -2/3],
    [ 2/3,  2/3,  2/3],
    [ 1/6,  1/6,  1/6]
]

// A^T: Output transform (4×4)
At = [
    [1, 1, 1, 1],
    [0, 1,-1, 2],
    [0, 1, 1, 4],
    [0, 1,-1, 8]
]
```

### 实现步骤

1. **Weight Transform**: `W = G @ weight @ G^T` (3×3 → 4×4 tiles)
   - 可预处理并缓存，不影响forward性能

2. **Input Transform**: `V = B^T @ input_tile @ B` (对每个4×4输入tile)
   - 每tile需要4×4输入，产生4×4输出

3. **Element-wise Multiplication**: `M = V * W`
   - 最耗时，可用Tensor Core加速

4. **Output Transform**: `Y = A^T @ M @ A`
   - 得到4×4输出tile

### Tile划分策略

```
输入: [N, C, H, W]
输出: [N, out_C, H-2, W-2] (3x3, pad=1, stride=1)

- 输入tile: 4×4 → 输出4×4
- Tile overlap: 每tile有2像素重叠 (3x3 kernel边界)
- 有效输出: 每tile产生4×4输出像素
- Tile数量: ceil((H-2)/4) × ceil((W-2)/4)
```

### 内存需求

```
额外内存:
- Weight transformed: out_C × C × 16 (4×4 tiles per weight)
- Input tiles buffer: N × C × num_tiles × 16
- Intermediate buffer: N × out_C × num_tiles × 16
```

### Kernel设计

```cpp
// conv2d_winograd.cu

// Weight transform (可预处理)
__global__ void winograd_weight_transform_kernel(
    const float* weight, float* weight_transformed,
    int out_C, int C);

// Input tile transform
__global__ void winograd_input_transform_kernel(
    const float* input, float* input_tiles,
    int N, int C, int H, int W, int num_tiles_h, int num_tiles_w);

// Element-wise multiplication (可使用Tensor Core)
__global__ void winograd_elementwise_kernel(
    const float* input_tiles, const float* weight_tiles,
    float* intermediate, int N, int C, int out_C, int num_tiles);

// Output transform
__global__ void winograd_output_transform_kernel(
    const float* intermediate, float* output,
    int N, int out_C, int out_H, int out_W);

// Forward function
void cuda_conv2d_winograd_forward(
    const float* input, const float* weight, const float* bias,
    float* output, int N, int C, int H, int W, int out_C);

// Backward function (需要推导反向变换)
void cuda_conv2d_winograd_backward(
    const float* grad_out, const float* input, const float* weight,
    float* grad_input, float* grad_weight, float* grad_bias, ...);
```

---

## FP16/Tensor Core设计

### 数据类型转换

```cpp
// half_utils.h
#include <cuda_fp16.h>

struct HalfConverter {
    // CPU侧转换
    static half* float_to_half(const float* data, size_t size);
    static float* half_to_float(const half* data, size_t size);
};

// GPU侧批量转换kernel
__global__ void float_to_half_kernel(const float* in, half* out, size_t n);
__global__ void half_to_float_kernel(const half* in, float* out, size_t n);
```

### Tensor Core WMMA API

```cpp
// tensor_core.cu
#include <mma.h>

using namespace nvcuda::wmma;

// WMMA tile: 16×16×16 (FP16输入, FP32累加)

__global__ void tensor_core_matmul_kernel(
    const half* A, const half* B, float* C,
    int M, int N, int K) {
    
    // 1. load_matrix_sync: 加载16×16 tiles到fragment
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    load_matrix_sync(a_frag, A + ...);
    load_matrix_sync(b_frag, B + ...);
    fill_fragment(c_frag, 0.0f);
    
    // 2. mma_sync: Tensor Core执行矩阵乘
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 3. store_matrix_sync: 存储结果
    store_matrix_sync(C + ..., c_frag, N, row_major);
}

// 封装函数
void cuda_matmul_tensor_core(
    const half* A, const half* B, float* C,
    size_t M, size_t N, size_t K);
```

### Conv2d with Tensor Core

```cpp
// conv2d_tensor_core.cu

void cuda_conv2d_tensor_core_forward(
    const float* input, const float* weight, float* output, ...) {
    
    // 1. im2col (保持FP32, 后转换为FP16)
    im2col_kernel<<<...>>>(input, col_buffer_fp32, ...);
    
    // 2. 转换为FP16
    float_to_half_kernel<<<...>>>(col_buffer_fp32, col_buffer_fp16, ...);
    float_to_half_kernel<<<...>>>(weight_fp32, weight_fp16, ...);
    
    // 3. Tensor Core matmul
    tensor_core_matmul_kernel<<<...>>>(weight_fp16, col_buffer_fp16, output_fp32, ...);
}
```

### 精度策略

```
输入: FP16 (存储节省, 带宽优化)
累加器: FP32 (避免精度损失累积)
输出: FP32 (保持训练稳定性)

FP16范围: ±65504, 精度约3位有效数字
对于CNN训练，FP16通常足够，但需要关注:
- 梯度缩放 (防止梯度溢出)
- Loss scaling (PyTorch AMP的标准做法)
```

---

## Kernel融合设计

### 融合模式

```
模式1: Conv → ReLU (单kernel)
模式2: Conv → ReLU → MaxPool (单kernel) ← 主要目标
模式3: Conv → Bias → ReLU (单kernel)
```

### 内存流量对比

```
分离模式:
  Conv输出 → 全局内存 → ReLU读取 → 全局内存 → Pool读取 → 全局内存
  全局内存写入: 3次

融合模式:
  Conv → ReLU (寄存器) → Pool (寄存器/warp) → 全局内存
  全局内存写入: 1次
  内存流量减少: ~3x
```

### Fused Kernel设计

```cpp
// conv2d_fused.cu

__global__ void conv2d_relu_pool_fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W,
    int out_C, int pool_size, int pool_stride, ...) {
    
    // Shared memory: 输入窗口 + 权重窗口
    __shared__ float input_tile[TILE_H + 2][TILE_W + 2];  // +2 for 3x3 kernel
    __shared__ float weight_tile[16][3][3];
    
    // 每thread计算一个conv输出像素
    int oc = blockIdx.y;
    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 1. 加载输入到shared memory
    cooperative_load_input(input_tile, input, ...);
    cooperative_load_weight(weight_tile, weight, oc, ...);
    __syncthreads();
    
    // 2. 计算卷积
    float conv_val = compute_conv_value(input_tile, weight_tile, ...);
    
    // 3. Bias + ReLU (inline, 无额外内存)
    conv_val += bias[oc];
    conv_val = fmaxf(0.0f, conv_val);  // ReLU
    
    // 4. MaxPool (warp内协作)
    // 每4×4 conv输出对应2×2 pool输出
    // warp内4个线程协作找最大值
    float pool_max = warp_reduce_max(conv_val, pool_pos);
    
    // 5. 写入输出 (只写一次全局内存!)
    if (is_pool_leader_thread) {
        output[pool_out_idx] = pool_max;
    }
}
```

### Warp-Level MaxPool

```cpp
// 2×2 pool: 每4线程协作
__device__ float warp_reduce_max(float val, int pool_pos) {
    // 使用shfl指令在warp内交换并比较
    // pool_pos = 0,1,2,3 对应2×2窗口的四个位置
    
    // Lane mask: 同一pool窗口的4个lane
    int lane_mask = get_pool_lane_mask(pool_pos);
    
    // 交换并比较
    float other1 = __shfl_xor_sync(lane_mask, val, 1);
    float other2 = __shfl_xor_sync(lane_mask, val, 2);
    float other3 = __shfl_xor_sync(lane_mask, val, 3);
    
    return fmaxf(fmaxf(fmaxf(val, other1), other2), other3);
}
```

### Backward处理

融合forward的backward需要拆分处理：

```cpp
void cuda_conv2d_relu_pool_backward(...) {
    // 1. grad_pool: scatter grad_out到pool窗口
    pool_backward_scatter_kernel<<<...>>>(grad_out, grad_relu, ...);
    
    // 2. grad_relu: grad * (output > 0)
    relu_backward_kernel<<<...>>>(grad_relu, relu_output, grad_conv, ...);
    
    // 3. grad_conv: 正常conv backward (可使用Winograd backward)
    cuda_conv2d_winograd_backward(grad_conv, input, weight, ...);
}
```

---

## 性能测试设计

### 测试配置矩阵

```python
test_configs = [
    # Winograd tests
    {
        'op': 'conv2d',
        'N': 64, 'C': 16, 'H': 28, 'W': 28, 'out_C': 32, 'K': 3,
        'stride': 1, 'pad': 1,
        'impl': ['im2col', 'winograd']
    },
    {
        'op': 'conv2d',
        'N': 64, 'C': 32, 'H': 14, 'W': 14, 'out_C': 64, 'K': 3,
        'impl': ['im2col', 'winograd', 'tensor_core']
    },
    
    # Tensor Core matmul tests
    {
        'op': 'matmul',
        'M': 512, 'N': 512, 'K': 512,
        'impl': ['tiled_fp32', 'tensor_core_fp16']
    },
    {
        'op': 'matmul',
        'M': 1024, 'N': 1024, 'K': 1024,
        'impl': ['tiled_fp32', 'tensor_core_fp16']
    },
    
    # Fused kernel tests
    {
        'op': 'conv_relu_pool',
        'N': 64, 'C': 16, 'H': 28, 'W': 28, 'out_C': 32, 'K': 3,
        'pool_size': 2, 'pool_stride': 2,
        'impl': ['separate', 'fused']
    },
    
    # Full training throughput
    {
        'op': 'training',
        'epochs': 5,
        'batch_size': 64,
        'impl': ['current', 'optimized', 'pytorch']
    },
]
```

### 对比基准

| 实现 | 描述 |
|------|------|
| current | 当前im2col + GEMM实现 |
| winograd | Winograd F(4×4, 3×3) |
| tensor_core_fp16 | FP16 + Tensor Core |
| fused | Conv→ReLU→Pool融合 |
| optimized | 组合所有优化 |
| pytorch | PyTorch cuDNN后端 |

### 报告格式

```
算子: conv2d (N=64, C=16, H=28, K=3)

| 实现 | 时间(ms) | GFLOPS | vs Current | vs PyTorch |
|------|----------|---------|------------|------------|
| current | 25.0 | 763 | 1.0x | 0.02x |
| winograd | 10.0 | 1908 | 2.5x | 0.05x |
| tensor_core | 8.0 | 2385 | 3.1x | 0.06x |
| fused | 15.0 | 1275 | 1.67x | 0.03x |
| optimized | 5.0 | 3815 | 5.0x | 0.10x |
| pytorch | 0.5 | 38150 | 50x | 1.0x |
```

---

## 实现优先级

### Phase 1: Winograd算法 (最高收益，中等复杂度)

1. 实现forward变换kernels
2. 实现backward变换kernels
3. 测试正确性
4. 性能测试

### Phase 2: FP16/Tensor Core (高收益，高复杂度)

1. FP16转换工具
2. Tensor Core matmul kernel
3. Conv2d with Tensor Core
4. 精度测试 + 训练稳定性测试

### Phase 3: Kernel融合 (中等收益，高复杂度)

1. Conv→ReLU融合
2. Conv→ReLU→Pool融合
3. Backward拆分实现
4. 性能测试

---

## 风险与挑战

| 技术 | 风险 | 缓解策略 |
|------|------|----------|
| Winograd | 边界处理 (H/W不是4的倍数) | Padding + 特殊边界kernel |
| Tensor Core | FP16精度损失 | FP32累加 + 梯度缩放 |
| Kernel融合 | Shared memory限制 | 分块策略 + 多级融合 |
| Backward | 反向传播复杂 | 保持backward分离，forward融合 |

---

## 预期成果

| 指标 | 当前 | 目标 | 说明 |
|------|------|------|------|
| 训练吞吐量 | 1000 samples/s | 5000-15000 samples/s | 5-15x加速 |
| Conv2d GFLOPS | 763 | 1900-3800 | Winograd + Tensor Core |
| vs PyTorch差距 | 42x | 3-8x | 显著缩小但仍有差距 |

---

*设计日期: 2026-04-29*