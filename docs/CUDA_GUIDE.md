# CUDA 编程要点

## 线程层次结构

```
Grid
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   ├── Thread 0
│   └── ...
└── ...
```

- **Grid**: 整个 GPU 上的线程集合
- **Block**: 一组线程，共享 shared memory
- **Thread**: 最小执行单元

## 索引计算

```cpp
// 全局线程索引
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 线程索引
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

## 内存类型

| 类型 | 位置 | 访问速度 | 作用域 |
|------|------|---------|-------|
| Register | GPU | 最快 | 单线程 |
| Shared Memory | GPU | 快 | Block 内 |
| Local Memory | GPU | 慢 | 单线程 |
| Global Memory | GPU | 最慢 | 全局 |

## Kernel 示例

```cpp
__global__ void my_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

// 调用
int block_size = 256;
int num_blocks = (size + block_size - 1) / block_size;
my_kernel<<<num_blocks, block_size>>>(input, output, size);
```

## 常见错误

### 1. 线程越界
```cpp
// 错误: 可能访问越界
output[idx] = input[idx] * 2.0f;

// 正确: 先检查
if (idx < size) {
    output[idx] = input[idx] * 2.0f;
}
```

### 2. 忘记检查 CUDA 错误
```cpp
// 错误
cudaMalloc(&ptr, size);

// 正确
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

### 3. 内存拷贝方向错误
```cpp
// 错误
cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);

// 正确
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
```

## 性能优化

### 1. 合并内存访问
```cpp
// 好的访问模式 (coalesced)
__global__ void good_access(const float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = input[idx];    // 相邻线程访问相邻内存
    output[idx] = val * 2.0f;
}
```

### 2. 使用 Shared Memory
```cpp
__global__ void shared_mem_kernel(float* data) {
    __shared__ float cache[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cache[threadIdx.x] = data[idx];

    __syncthreads();

    // 使用缓存的数据
    float result = cache[threadIdx.x] * 2.0f;
}
```

### 3. 避免 Bank Conflict
- Shared memory 有 32 个 bank
- 相邻线程访问相邻地址会导致 bank conflict
- 使用 padding 避免冲突

## CUDA 数学函数

| CPU 函数 | CUDA 函数 | 说明 |
|---------|----------|------|
| std::abs | fabsf | 绝对值 |
| std::max | fmaxf | 最大值 |
| std::min | fminf | 最小值 |
| std::exp | expf | 指数 |
| std::log | logf | 对数 |
| std::sqrt | sqrtf | 平方根 |

**注意**: 在 CUDA kernel 中使用 `std::` 函数可能导致性能问题或编译错误，应使用对应的 CUDA 版本。

## 原子操作

```cpp
__global__ void atomic_add(float* data, float val) {
    atomicAdd(data, val);
}
```

常见原子操作：
- `atomicAdd()` - 浮点加法
- `atomicExch()` - 交换
- `atomicCAS()` - 比较并交换

## 流 (Streams)

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

my_kernel<<<num_blocks, block_size, 0, stream>>>(input, output, size);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

---

## 高级优化技术

### Tensor Core (WMMA API)

Tensor Core 是 NVIDIA GPU 上的专用矩阵计算单元，可加速 FP16 矩阵乘法。

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// 16×16×16 Tensor Core 操作
__global__ void tensor_core_matmul(const half* A, const half* B, half* C, 
                                    int M, int N, int K) {
    // 定义 WMMA fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;
    
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 16;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    
    // 初始化 accumulator
    fill_fragment(c_frag, 0.0f);
    
    // 遍历 K 维，每次处理 16 个元素
    for (int k = 0; k < K; k += 16) {
        // 加载 A 和 B tiles
        load_matrix_sync(a_frag, A + warp_row * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warp_col * 16, N);
        
        // Tensor Core 矩阵乘法
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, mem_row_major);
}
```

**注意事项**:
- WMMA 操作以 warp (32 threads) 为单位，不是单线程
- Tensor Core 需要 FP16 输入，累积结果可以是 FP16 或 FP32
- Tesla T4: FP16 峰值 65 TFLOPS，FP32 峰值 8.1 TFLOPS

### FP16 数据类型

```cpp
// CUDA half 类型
#include <cuda_fp16.h>

// FP32 → FP16 转换
__device__ half float_to_half(float f) {
    return __float2half(f);
}

// FP16 → FP32 转换
__device__ float half_to_float(half h) {
    return __half2float(h);
}

// 向量化转换 (一次转换多个)
__global__ void convert_fp32_to_fp16(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}
```

---

### Winograd 卷积算法

Winograd 算法通过数学变换减少卷积乘法次数。

**F(2×2, 3×3) 变换矩阵** (wincnn 标准):

```
A^T = [[1, 1, 1, 0],     输出变换: Y = A^T @ M @ A
       [0, 1, -1, 1]]

G   = [[1, 0, 0],        权重变换: W = G @ w @ G^T
       [1/2, 1/2, 1/2],
       [1/2, -1/2, 1/2],
       [0, 0, 1]]

B^T = [[1, 0, -1, 0],    输入变换: V = B^T @ U @ B
       [0, 1, 1, 0],
       [0, -1, 1, 0],
       [0, 1, 0, -1]]
```

**实现步骤**:

```cpp
// 1. 输入变换 (每 batch)
__global__ void winograd_input_transform(...) {
    // 读取 4×4 输入 tile U
    // 计算 V = B^T @ U @ B
    // 存储 V 到 temp buffer
}

// 2. 权重变换 (预计算，一次)
__global__ void winograd_weight_transform(...) {
    // 读取 3×3 权重 w
    // 计算 W = G @ w @ G^T
    // 存储 W 到 temp buffer
}

// 3. 元素乘法
__global__ void winograd_elementwise(...) {
    // M = V ⊙ W (逐元素乘)
}

// 4. 输出变换
__global__ void winograd_output_transform(...) {
    // Y = A^T @ M @ A
    // 从 4×4 M 提取 2×2 输出 Y
}
```

**性能收益**:
- 标准 3×3 卷积: 9 次乘法/输出像素
- Winograd F(2×2): 4 次乘法/输出像素 (理论 2.25x)
- cuDNN F(6×6): ~5x 加速

---

### Kernel Fusion

将多个 kernel 合并为一个，减少 kernel launch 和内存读写开销。

```cpp
// 分离 kernel (慢)
conv2d_kernel<<<...>>>(input, weight, conv_out);   // ~5μs launch
cudaDeviceSynchronize();                           // 同步开销
relu_kernel<<<...>>>(conv_out, relu_out);          // ~5μs launch
cudaDeviceSynchronize();

// 融合 kernel (快)
fused_conv_relu_kernel<<<...>>>(input, weight, output);  // ~5μs launch (一次)
cudaDeviceSynchronize();

// 融合 kernel 实现
__global__ void fused_conv_relu_kernel(...) {
    // 计算卷积 (结果在 register 或 shared memory)
    float conv_val = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                conv_val += input[...] * weight[...];
            }
        }
    }
    
    // 直接应用 ReLU，无需写入全局内存再读取
    output[idx] = conv_val > 0 ? conv_val : 0.0f;
}
```

**收益分析**:
- Kernel launch: ~5-10μs/次 → 减少一次 = ~5μs
- 内存读写: Conv_out 写入 + ReLU 读取 → 零中间内存
- 实际加速: ~1.2x (取决于 conv 计算 vs launch 开销比例)

---

### Warp Shuffle 指令

在 warp (32 threads) 内直接传递数据，无需 shared memory。

```cpp
// Warp 内 reduction
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp 内广播
__device__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}
```

**应用场景**:
- Softmax max/sum 计算
- Block 内部分结果汇总
- 避免 shared memory bank conflict

---

## 性能分析工具

### nvprof (命令行)

```bash
# 分析 kernel 时间
nvprof ./bin/benchmark

# 详细分析
nvprof --print-gpu-trace ./bin/benchmark
```

### nsys (新一代工具)

```bash
# 生成分析报告
nsys profile --stats=true ./bin/benchmark

# GUI 分析
nsys-ui ./report.nsys-rep
```

### CUDA Event 计时

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<...>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float elapsed_ms;
cudaEventElapsedTime(&elapsed_ms, start, stop);
printf("Kernel time: %.2f ms\n", elapsed_ms);
```

---

## 常见问题解决

### Bank Conflict

```cpp
// 问题: 相邻线程访问相邻地址导致 bank conflict
__shared__ float data[256];
data[threadIdx.x] = ...;  // OK: 每个 bank 一线程
data[threadIdx.x * 32] = ...;  // BAD: 所有线程访问同一 bank

// 解决: Padding
__shared__ float data[256 + 8];  // 避开 bank 冲突
```

### Occupancy 优化

```cpp
// 计算最大 occupancy
int max_threads_per_block;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, my_kernel, 
                                                block_size, dynamic_smem);

// 调整 block size 以最大化 occupancy
// 一般: 128, 256, 512 threads/block
```

### 内存对齐

```cpp
// 对齐访问更高效
struct AlignedData {
    float a;
    float b;
    float c;
    float d;  // 16 bytes aligned
};

// 或使用对齐属性
struct __align__(16) MyStruct { ... };
```

---

## 参考资源

- [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Tensor Core 指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-core-programming)
- [Winograd 论文](https://arxiv.org/abs/1509.09308)
- [wincnn 矩阵生成器](https://github.com/andravin/wincnn)
- [CUDA 最佳实践](https://docs.nvidia.com/cuda/cuda-best-practices-guide/)

---

*文档版本: 1.3.0 - 2026-04-30*
