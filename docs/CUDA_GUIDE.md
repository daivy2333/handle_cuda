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
