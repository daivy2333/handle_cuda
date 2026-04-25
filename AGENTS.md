# AGENTS.md

## 项目概述

CUDA 深度学习算子实现项目，用于学习和实践 CUDA 编程。

## 技术栈

- CUDA C/C++ (C++17, CUDA 17)
- CMake
- GoogleTest
- PyTorch (用于验证)

## 构建和测试

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make run_tests
```

## 算子实现清单

### 核心文件

| 文件 | 描述 |
|------|------|
| `include/cuda_ops.h` | 所有算子声明 |
| `include/cuda_util.h` | CUDA 工具宏和辅助函数 |
| `src/*.cu` | 各算子实现 |

### 已实现

- `cuda_matmul` - 矩阵乘法，支持转置
- `cuda_bias_add` - 偏置加法
- `cuda_relu` / `cuda_relu_backward` - ReLU 激活及反向
- `cuda_softmax` / `cuda_softmax_backward` - Softmax 及反向
- `cuda_conv2d` - 2D 卷积
- `cuda_maxpool2d` / `cuda_maxpool2d_backward` - MaxPool 及反向

## 添加新算子的步骤

1. 在 `include/cuda_ops.h` 中添加函数声明
2. 在 `src/` 下创建对应的 `.cu` 文件
3. 在 `tests/` 下创建对应的测试文件
4. 在 `tests/CMakeLists.txt` 中添加测试目标
5. 更新 `README.md` 中的算子列表

## 代码风格

- 使用 `CUDA_CHECK` 宏检查 CUDA 错误
- 使用 `__global__` 定义 kernel
- 使用 `dim3` 配置线程块和网格
- 使用 `CudaBuffer` 管理设备内存
- 避免在 kernel 中使用 `std::` 命名空间（用 `fmaxf` 替代 `std::fmax`）

## 测试验证

每个算子的测试必须：
1. 与 CPU 参考实现对比（误差 < 1e-5）
2. 测试边界情况（空数组、单元素、大数组）
3. 测试反向传播（如适用）

## 性能目标

- MatMul: 目标 GFLOPS > 1000（Tesla T4）
- ReLU: < 0.1 ms for 10M elements
- Conv2d: 对标 PyTorch torch.conv2d

## 目录结构

```
cuda_dl_ops/
├── CMakeLists.txt
├── include/
│   ├── cuda_ops.h      # Public API
│   └── cuda_util.h     # Internal utilities
├── src/                # Implementation (.cu files)
├── tests/              # GoogleTest tests
├── scripts/            # Python benchmark scripts
└── docs/               # Documentation
```

## 注意事项

1. **不要修改 `include/cuda_util.h`** 中的 `CUDA_CHECK` 宏
2. **所有 kernel 必须检查 `cudaGetLastError()`**
3. **测试必须使用随机数据**，避免 hardcoded 测试用例
4. **使用 `device_to_device` 进行设备内拷贝**，`host_to_device`/`device_to_host` 用于主机设备间拷贝
