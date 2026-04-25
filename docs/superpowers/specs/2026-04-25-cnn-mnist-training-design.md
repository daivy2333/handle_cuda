# CNN MNIST 训练系统设计

## 1. 项目概述

### 目标
利用已实现的 CUDA 深度学习算子构建一个完整的 CNN 训练系统，在 MNIST 数据集上训练手写数字分类模型，并与 PyTorch 实现进行正确性、收敛性和性能对比。

### 成功标准
- 正确性：单 pass 输出与 PyTorch 误差 < 1e-4
- 收敛性：10 epochs 后测试准确率 > 95%（接近 PyTorch）
- 性能：训练时间与 PyTorch 相当（差距 < 50%）

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                     Python 层                        │
│                                                      │
│   train_mnist.py          compare_pytorch.py        │
│   ├─ 数据加载 (numpy)     ├─ 正确性验证             │
│   ├─ 训练循环             ├─ 收敛对比               │
│   ├─ 模型封装             ├─ 性能测试               │
│   └─ 评估/可视化          └─ 结果输出               │
│                                                      │
│                      cuda_ops.py                     │
│                    (ctypes binding)                  │
└─────────────────────────────────────────────────────┘
                         ↓ ctypes
┌─────────────────────────────────────────────────────┐
│                     CUDA 层                          │
│                                                      │
│   已有算子              新增组件                     │
│   ├─ MatMul            ├─ CrossEntropyLoss          │
│   ├─ Conv2d            ├─ CrossEntropyLossBackward  │
│   ├─ MaxPool2d         ├─ SGDUpdate                 │
│   ├─ ReLU              ├─ BatchMatMul (可选)        │
│   ├─ Softmax           └─ Flatten                   │
│   ├─ BiasAdd                                          │
│   └─ Dropout                                          │
│                                                      │
│                      cuda_ops.h                      │
│                    (C API 导出)                      │
└─────────────────────────────────────────────────────┘
```

---

## 3. CNN 网络结构

采用简化版 LeNet-5：

```
Layer Structure:
┌─────────────────────────────────────────────────────────┐
│ Input: [batch, 1, 28, 28]                               │
│                                                         │
│ Conv1: Conv2d(1→6, 5x5, pad=2) → ReLU → MaxPool(2x2)   │
│        Output: [batch, 6, 14, 14]                      │
│                                                         │
│ Conv2: Conv2d(6→16, 5x5) → ReLU → MaxPool(2x2)         │
│        Output: [batch, 16, 5, 5] = 400 features        │
│                                                         │
│ Flatten: [batch, 400]                                   │
│                                                         │
│ FC1: Linear(400→120) → ReLU                            │
│                                                         │
│ FC2: Linear(120→84) → ReLU                             │
│                                                         │
│ FC3: Linear(84→10)                                     │
│                                                         │
│ Output: CrossEntropyLoss (Softmax + NLL)               │
│         [batch, 10] logits                             │
└─────────────────────────────────────────────────────────┘

参数统计:
- Conv1: 6×1×5×5 = 150 weights + 6 bias = 156
- Conv2: 16×6×5×5 = 2400 weights + 16 bias = 2416
- FC1: 400×120 = 48000 weights + 120 bias = 48120
- FC2: 120×84 = 10080 weights + 84 bias = 10164
- FC3: 84×10 = 840 weights + 10 bias = 850
- Total: ~62k parameters
```

---

## 4. 新增 CUDA 组件

### 4.1 CrossEntropyLoss

**Forward**:
```
输入: logits [batch, 10], targets [batch] (class indices)
输出: loss [scalar]

公式:
  softmax_logits = softmax(logits)
  log_softmax = log(softmax_logits)
  loss = -mean(log_softmax[i, targets[i]] for i in batch)
```

**Backward**:
```
输入: grad_out [scalar] (= 1.0), logits, targets
输出: grad_logits [batch, 10]

公式:
  grad_logits = softmax_logits - one_hot(targets)
  grad_logits /= batch_size
```

### 4.2 SGDUpdate

**公式**:
```
weight = weight - learning_rate * grad_weight
bias = bias - learning_rate * grad_bias
```

实现：简单的 element-wise kernel，可复用 BiasAdd 的模式。

### 4.3 Flatten

**Forward**:
```
输入: [batch, C, H, W]
输出: [batch, C*H*W]

实现: 内存重排，只需修改索引计算
```

**Backward**:
```
输入: grad_flat [batch, C*H*W]
输出: grad_input [batch, C, H, W]

实现: 反向重排
```

---

## 5. Python Binding 设计

### 5.1 ctypes 接口

```python
# cuda_ops.py

import ctypes
import numpy as np

class CUDAOps:
    def __init__(self):
        self.lib = ctypes.CDLL('./libcuda_ops.so')
        self._setup_functions()
    
    def conv2d(self, input, weight, bias, desc):
        # input: np.ndarray [N, C, H, W]
        # weight: np.ndarray [out_C, C, kH, kW]
        # Returns: np.ndarray [N, out_C, out_H, out_W]
        ...
    
    def matmul(self, A, B):
        # A: [M, K], B: [K, N]
        # Returns: [M, N]
        ...
    
    def relu(self, data):
        # inplace relu
        ...
    
    def softmax(self, input):
        # input: [batch, classes]
        # Returns: [batch, classes]
        ...
    
    def cross_entropy_loss(self, logits, targets):
        # logits: [batch, 10], targets: [batch] int
        # Returns: loss scalar, grad_logits [batch, 10]
        ...
    
    def sgd_update(self, param, grad, lr):
        # inplace: param -= lr * grad
        ...
```

### 5.2 模型封装

```python
# model.py

class SimpleCNN:
    def __init__(self):
        # 初始化权重 (random)
        self.conv1_weight = randn(6, 1, 5, 5)
        self.conv1_bias = zeros(6)
        ...
    
    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = cuda_ops.conv2d(x, self.conv1_weight, self.conv1_bias, ...)
        x = cuda_ops.relu(x)
        x = cuda_ops.maxpool2d(x, ...)
        ...
        return logits
    
    def backward(self, grad_logits):
        # 反向传播，更新 grad_weight, grad_bias
        ...
    
    def update(self, lr):
        # SGD 更新所有参数
        cuda_ops.sgd_update(self.conv1_weight, self.grad_conv1_weight, lr)
        ...
```

---

## 6. 训练流程

```python
# train_mnist.py

# 1. 加载 MNIST 数据
train_images, train_labels = load_mnist('train')
test_images, test_labels = load_mnist('test')

# 2. 初始化模型
model = SimpleCNN()
optimizer_lr = 0.01
batch_size = 64

# 3. 训练循环
for epoch in range(10):
    for batch in range(num_batches):
        # 获取 batch 数据
        x = train_images[batch*64:(batch+1)*64]
        y = train_labels[batch*64:(batch+1)*64]
        
        # Forward
        logits = model.forward(x)
        loss, grad_logits = cuda_ops.cross_entropy_loss(logits, y)
        
        # Backward
        model.backward(grad_logits)
        
        # Update
        model.update(optimizer_lr)
    
    # Evaluate
    test_acc = evaluate(model, test_images, test_labels)
    print(f"Epoch {epoch}: loss={loss:.4f}, test_acc={test_acc:.2%}")
```

---

## 7. 对比方案

### 7.1 正确性验证

```python
# compare_correctness.py

# 固定输入
x = np.random.randn(64, 1, 28, 28).astype(np.float32)
y = np.random.randint(0, 10, 64)

# 我们的实现
our_logits = our_model.forward(x)

# PyTorch 实现
torch_logits = torch_model(x)

# 对比
diff = np.abs(our_logits - torch_logits.detach().numpy())
print(f"Max diff: {diff.max():.6f}")
assert diff.max() < 1e-4
```

### 7.2 收敛对比

```python
# compare_convergence.py

# 相同超参数训练两个模型
our_history = train_our_model(lr=0.01, epochs=10)
torch_history = train_torch_model(lr=0.01, epochs=10)

# 绘制曲线
plot_comparison(our_history, torch_history)
```

### 7.3 性能对比

```python
# compare_performance.py

# 训练时间
our_time = time_training(our_model, train_data)
torch_time = time_training(torch_model, train_data)

print(f"Our time: {our_time:.2f}s")
print(f"PyTorch time: {torch_time:.2f}s")
print(f"Ratio: {our_time/torch_time:.2f}x")
```

---

## 8. 文件结构

```
handle_cuda/
├── src/
│   ├── ... (已有算子)
│   ├── cross_entropy.cu      # 新增
│   ├── sgd_update.cu         # 新增
│   ├── flatten.cu            # 新增
│   └── cuda_ops_export.cu    # C API 导出
├── include/
│   └── cuda_ops.h            # 更新 API
├── python/
│   ├── cuda_ops.py           # ctypes binding
│   ├── model.py              # CNN 模型封装
│   ├── train_mnist.py        # 训练脚本
│   ├── compare_pytorch.py    # 对比脚本
│   └── mnist_data.py         # 数据加载
├── tests/
│   └── test_training.cpp     # 训练相关测试
└── docs/
    └── TRAINING_GUIDE.md     # 使用文档
```

---

## 9. 实现优先级

### Wave 1: 核心 CUDA 组件
- CrossEntropyLoss forward/backward
- SGDUpdate kernel
- C API 导出

### Wave 2: Python 绑定
- ctypes binding 封装
- 数据加载 (MNIST)
- 模型封装类

### Wave 3: 训练和对比
- 训练脚本
- PyTorch 对比脚本
- 可视化

---

## 10. 风险和限制

### 已知限制
1. 无 BatchNorm：收敛可能比 PyTorch 慢
2. 无学习率调度：固定 lr
3. 无数据增强：原始 MNIST
4. 单精度 FP32：无 FP16 支持

### 风险点
1. **梯度累积正确性**：backward pass 需要仔细验证
2. **内存管理**：Python 和 CUDA 内存交互需注意生命周期
3. **数值稳定性**：Softmax + Log 需要防止 overflow

---

## 11. 验收测试

| 测试 | 标准 |
|------|------|
| CrossEntropyLoss forward | 与 PyTorch 误差 < 1e-5 |
| CrossEntropyLoss backward | 与 PyTorch 梯度误差 < 1e-5 |
| 训练收敛 | 10 epochs 后 test_acc > 95% |
| 性能对比 | 训练时间差距 < 50% |
| 内存检查 | 无泄漏，无非法访问 |

---

*Design created: 2026-04-25*