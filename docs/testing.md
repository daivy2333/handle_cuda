# 测试指南

## 测试框架

使用 GoogleTest 作为单元测试框架。

## 运行测试

```bash
# 构建后运行所有测试
cd build
make run_tests

# 或直接运行单个测试
./bin/test_matmul
./bin/test_relu
./bin/test_bias_add
./bin/test_softmax
./bin/test_conv2d
./bin/test_maxpool2d
```

## 测试覆盖

| 算子 | 基础测试 | 边界测试 | 反向测试 |
|------|---------|---------|---------|
| MatMul | ✅ | ✅ (转置, 大矩阵) | - |
| BiasAdd | ✅ | ✅ (单行, 大矩阵) | - |
| ReLU | ✅ | ✅ (全正, 全负) | ✅ |
| Softmax | ✅ | ✅ (和为1, 非负) | - |
| Conv2d | ✅ | ✅ (无padding) | - |
| MaxPool2d | ✅ | ✅ (padding, batch) | - |

## 测试方法

### 正确性验证

每个测试都包含：
1. **CPU 参考实现** - 用 C++ 计算期望结果
2. **CUDA 实现** - 调用我们的算子
3. **误差比较** - 验证误差在可接受范围内

```cpp
float relative_error = abs(cuda_result - cpu_result) / (abs(cpu_result) + 1e-6);
EXPECT_LT(relative_error, 1e-5f);
```

### 误差标准

| 算子 | 误差阈值 |
|------|---------|
| ReLU | 0 (精确) |
| BiasAdd | 1e-6 |
| Softmax | 1e-5 |
| MatMul | 1e-4 |
| Conv2d | 1e-4 |
| MaxPool2d | 0 (精确) |

### 边界情况

- **空数组**: size = 0
- **单元素**: size = 1
- **大数组**: size = 10M+
- **特殊值**: 全0, 全正, 全负

## 添加新测试

1. 在 `tests/` 创建 `test_xxx.cpp`
2. 继承 `::testing::Test`
3. 在 `SetUp()` 中初始化 CUDA

```cpp
class MyOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
};

TEST_F(MyOpTest, Basic) {
    // 测试代码
}
```

4. 在 `tests/CMakeLists.txt` 添加

```cmake
add_executable(test_xxx test_xxx.cpp)
target_link_libraries(test_xxx cuda_ops GTest::gtest GTest::gtest_main pthread)
```
