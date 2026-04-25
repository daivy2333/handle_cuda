# 实现细节

## MatMul (矩阵乘法)

### 思路
朴素 GEMM 实现，按 block 分发线程，每个线程计算 C[i][j]。

### Kernel 设计
- Block size: 16x16
- Grid size: `(N+15)/16 x (M+15)/16`
- 每个线程计算一个输出元素

### 计算方式

对于 C = A × B：
- A: M×K, B: K×N, C: M×N
- 线程 (row, col) 计算 C[row][col]
- 遍历 k 计算 A[row][k] * B[k][col] 的和

### 转置支持

通过 `MatMulDesc.transpose_a/b` 支持：
- 转置 A: A^T × B
- 转置 B: A × B^T
- 转置两者: A^T × B^T

### 优化方向
- [ ] 使用 shared memory 缓存 A 的行
- [ ] 使用 cublas 对比性能
- [ ] 支持 tensor core
- [ ] 添加 FP16 支持

## ReLU

### 思路
Element-wise 操作，每个线程处理一个元素。

### Kernel 设计
- Block size: 256
- Grid size: `(size+255)/256`

### 正向
```cuda
data[idx] = fmaxf(0.0f, data[idx]);
```

### 反向
```cuda
grad_in[idx] = forward_input[idx] > 0.0f ? grad_out[idx] : 0.0f;
```

## BiasAdd

### 思路
沿行方向广播偏置，每个线程处理一个元素。

### Kernel 设计
- Block size: 256
- Grid size: `(rows * cols + 255) / 256`

### 计算方式
```
output[row * cols + col] = input[row * cols + col] + bias[col]
```

## Softmax

### 思路
按 batch 并行，每个 batch 一个 block。

### Kernel 设计
- Grid size: `batch_size` (每个 batch 一个 block)
- Block size: 1

### 计算流程 (数值稳定)
1. 找每行最大值：`max_val = max(input)`
2. 计算指数和：`sum = sum(exp(input - max_val))`
3. 归一化：`output = exp(input - max_val) / sum`

### 反向传播
```
grad_in[i] = forward_output[i] * (grad_out[i] - sum(grad_out * forward_output))
```

## Conv2d

### 思路
朴素 Im2Col 实现，直接卷积。

### Kernel 设计
- Grid: `(out_H * out_W) x out_C x N`
- Block: 16x16

### 计算流程
对于每个输出位置 (n, oc, oh, ow)：
1. 遍历输入通道 ic 和 kernel 位置 kh, kw
2. 计算输入位置 ih = oh * stride_h + kh - pad_h
3. 计算输入位置 iw = ow * stride_w + kw - pad_w
4. 累加 input[n, ic, ih, iw] * weight[oc, ic, kh, kw]

### 优化方向
- [ ] 使用 shared memory
- [ ] 支持 groups
- [ ] 支持 dilation
- [ ] Im2Col + GEMM 实现

## MaxPool2d

### 思路
朴素实现，每个线程找 kernel 内的最大值并记录索引。

### Kernel 设计
- Grid: `(out_H * out_W) x C x N`
- Block: 16x16

### 正向
1. 遍历 kernel 内的所有位置
2. 找最大值并记录原始索引
3. 输出最大值，保存索引用于反向

### 反向
- 使用 `atomicAdd` 将梯度加到最大值的原始位置
- 因为多个输出可能共享同一个输入位置

### 注意事项
- 需要 `int*` 单独管理 indices 缓冲区
- 不能使用 `CudaBuffer` 因为它是 float 专用
