# CUDA Deep Learning Operators

用 CUDA 实现深度学习核心算子，作为 CUDA 编程的学习项目。

## 技术栈

| 组件 | 技术 |
|------|------|
| 核心语言 | CUDA C/C++ (C++17) |
| 构建工具 | CMake 3.18+ |
| 测试框架 | GoogleTest |
| 验证框架 | PyTorch (用于结果对比) |
| GPU | NVIDIA CUDA 11.5+ |

## 项目结构

```
cuda_dl_ops/
├── CMakeLists.txt           # 顶层构建配置
├── include/
│   ├── cuda_ops.h          # 公共 API (算子声明)
│   └── cuda_util.h         # 内部工具 (宏、辅助函数)
├── src/                    # 算子实现 (.cu 文件)
├── tests/                  # GoogleTest 单元测试
├── scripts/                # Python 验证脚本
└── docs/                   # 架构文档
```

## 已实现算子

| 算子 | 文件 | Forward | Backward |
|------|------|---------|----------|
| MatMul | matmul.cu | ✅ | - |
| BiasAdd | bias_add.cu | ✅ | - |
| ReLU | relu.cu | ✅ | ✅ |
| Softmax | softmax.cu | ✅ | ✅ |
| Conv2d | conv2d.cu | ✅ | - |
| MaxPool2d | maxpool2d.cu | ✅ | ✅ |

## 设计原则

1. **零外部依赖** - 除 CUDA/PyTorch 外不引入额外库
2. **单算子单文件** - 便于学习和维护
3. **测试驱动** - 每个算子都有对应的 GoogleTest 测试
4. **可验证** - 测试中包含与 PyTorch 结果的对比

## 目录结构说明

### `include/`

存放公共头文件：

- `cuda_ops.h` - 算子 API 声明，包含所有公开接口
- `cuda_util.h` - 工具宏和辅助类，如 `CUDA_CHECK`、`CudaBuffer`

### `src/`

存放算子实现，每个算子一个 `.cu` 文件：

- 每个文件实现一个或多个相关算子
- 使用 `__global__` 定义 kernel
- 使用 `dim3` 配置线程块和网格

### `tests/`

存放 GoogleTest 测试文件：

- `test_*.cpp` - 每个算子对应一个测试文件
- 测试验证正确性（与 CPU 参考实现对比）
- 测试边界情况（空数组、单元素、大数组）

### `scripts/`

存放 Python 验证脚本：

- `benchmark.py` - 性能对比脚本，与 PyTorch 对比

## 核心组件

### CUDA_CHECK 宏

用于检查 CUDA API 调用错误：

```cpp
CUDA_CHECK(cudaMalloc(&ptr, size));
```

### CudaBuffer 类

管理设备内存的 RAII 包装：

```cpp
CudaBuffer buffer(1024);  // 分配 1024 个 float
buffer.clear();           // 清零
```

### Helper 函数

| 函数 | 用途 |
|------|------|
| `host_to_device_async()` | 同步主机→设备复制 |
| `device_to_host()` | 设备→主机复制 |
| `device_to_device()` | 设备内复制 |
| `get_num_blocks()` | 计算 grid 大小 |

## 构建和测试

```bash
# 构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行所有测试
make run_tests

# 运行单个测试
./bin/test_relu
./bin/test_matmul
```

## 添加新算子

1. 在 `include/cuda_ops.h` 添加函数声明
2. 在 `src/` 创建对应的 `.cu` 文件
3. 在 `tests/` 创建对应的测试文件
4. 在 `tests/CMakeLists.txt` 添加测试目标
5. 更新 `README.md` 中的算子列表
