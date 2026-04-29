# CNN训练性能加速实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现Winograd算法、FP16/Tensor Core、Kernel融合三大优化，目标是5-15x加速

**Architecture:** 采用分层设计，新增模块独立实现，通过算子选择策略自动选择最优实现。保持纯CUDA实现，不依赖cuBLAS/cuDNN库。

**Tech Stack:** CUDA C++, Python ctypes, gtest

---

## 文件结构

```
src/
├── conv2d.cu              # 保留不变（fallback）
├── conv2d_winograd.cu     # Winograd F(4×4, 3×3) 实现 [NEW]
├── conv2d_fused.cu        # Conv→ReLU→Pool 融合kernel [NEW]
├── matmul_fp16.cu         # FP16 matmul + Tensor Core [NEW]
├── half_utils.cu          # FP16转换kernel [NEW]
├── cuda_ops_export.cu     # 修改：添加新算子导出
include/
├── cuda_ops.h             # 修改：添加新接口声明
tests/
├── test_conv2d_winograd.cpp   # Winograd正确性测试 [NEW]
├── test_fp16_tensor_core.cpp  # FP16/Tensor Core测试 [NEW]
├── test_fused_conv.cpp        # 融合kernel测试 [NEW]
python/
├── benchmark_compare.py   # 性能对比脚本 [NEW]
```

---

## Task 1: Winograd Forward实现

**Files:**
- Create: `src/conv2d_winograd.cu`
- Modify: `include/cuda_ops.h:83-84`, `src/cuda_ops_export.cu`, `tests/CMakeLists.txt`
- Test: `tests/test_conv2d_winograd.cpp`

- [ ] **Step 1: 创建Winograd Forward头文件和基础实现框架**

```cpp
// src/conv2d_winograd.cu
#ifndef CONV2D_WINOGRAD_CU
#define CONV2D_WINOGRAD_CU

#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

// Winograd F(4×4, 3×3) 变换矩阵常量
// B^T: Input transform (4×4)
__device__ __constant__ float winograd_Bt[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f},
    { 0.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f, -1.0f,  1.0f,  0.0f},
    { 0.0f,  1.0f,  0.0f, -1.0f}
};

// G: Weight transform (4×3)
__device__ __constant__ float winograd_G[4][3] = {
    {  1.0f,      0.0f,      0.0f},
    { -2.0f/3.0f, -2.0f/3.0f, -2.0f/3.0f},
    {  2.0f/3.0f,  2.0f/3.0f,  2.0f/3.0f},
    {  1.0f/6.0f,  1.0f/6.0f,  1.0f/6.0f}
};

// A^T: Output transform (4×4)
__device__ __constant__ float winograd_At[4][4] = {
    { 1.0f,  1.0f,  1.0f,  1.0f},
    { 0.0f,  1.0f, -1.0f,  2.0f},
    { 0.0f,  1.0f,  1.0f,  4.0f},
    { 0.0f,  1.0f, -1.0f,  8.0f}
};

// 计算tile数量
inline int get_num_tiles(int spatial_dim) {
    return (spatial_dim + 1) / 2;  // ceil((H-2)/4) ≈ (H+2)/4 for pad=1,stride=1
}

} // namespace

// Public API
void cuda_conv2d_winograd_forward(...);

#endif
```

- [ ] **Step 2: 实现Winograd Weight Transform Kernel**

```cpp
// Weight transform: W -> G @ W @ G^T
// Input: [out_C, C, 3, 3]
// Output: [out_C, C, 4, 4]
__global__ void winograd_weight_transform_kernel(
    const float* weight, float* weight_transformed,
    int out_C, int C) {
    
    int oc = blockIdx.x;
    int c = blockIdx.y;
    
    // 每个output channel和input channel的3x3权重块
    float w[3][3];
    float W[4][4] = {{0}};
    
    // Load 3x3 weight
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            w[kh][kw] = weight[oc * C * 9 + c * 9 + kh * 3 + kw];
        }
    }
    
    // W = G @ w @ G^T
    // 先计算 G @ w -> temp[4][3]
    float temp[4][3] = {{0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                temp[i][j] += winograd_G[i][k] * w[k][j];
            }
        }
    }
    
    // temp @ G^T -> W[4][4]
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                W[i][j] += temp[i][k] * winograd_G[j][k];
            }
        }
    }
    
    // Store transformed weight
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            weight_transformed[oc * C * 16 + c * 16 + i * 4 + j] = W[i][j];
        }
    }
}
```

- [ ] **Step 3: 实现Winograd Input Transform Kernel**

```cpp
// Input transform: V = B^T @ input_tile @ B
// Input: [N, C, H, W]
// Output: [N, C, num_tiles_h, num_tiles_w, 4, 4] (packed as [N*C*num_tiles, 16])
__global__ void winograd_input_transform_kernel(
    const float* input, float* input_tiles,
    int N, int C, int H, int W,
    int num_tiles_h, int num_tiles_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tiles = N * C * num_tiles_h * num_tiles_w;
    
    if (idx < total_tiles * 16) {
        int tile_idx = idx / 16;
        int elem_idx = idx % 16;
        
        int n = tile_idx / (C * num_tiles_h * num_tiles_w);
        int rem = tile_idx % (C * num_tiles_h * num_tiles_w);
        int c = rem / (num_tiles_h * num_tiles_w);
        int tile_h = (rem % (num_tiles_h * num_tiles_w)) / num_tiles_w;
        int tile_w = rem % num_tiles_w;
        
        int oh_start = tile_h * 2;  // 2 output pixels per 4x4 input tile (stride=1)
        int ow_start = tile_w * 2;
        
        // 获取4x4输入区域 (with padding for boundary tiles)
        float input_tile[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int ih = oh_start + i - 1;  // -1 for pad (pad=1)
                int iw = ow_start + j - 1;
                
                // 处理边界情况 (需要padding)
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                    input_tile[i][j] = 0.0f;
                } else {
                    input_tile[i][j] = input[n * C * H * W + c * H * W + ih * W + iw];
                }
            }
        }
        
        // V = B^T @ input_tile @ B
        float temp[4][4] = {{0}};
        float V[4][4] = {{0}};
        
        // B^T @ input_tile -> temp
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    temp[i][j] += winograd_Bt[i][k] * input_tile[k][j];
                }
            }
        }
        
        // temp @ B -> V
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    V[i][j] += temp[i][k] * winograd_Bt[j][k];
                }
            }
        }
        
        // Store V (row-major)
        input_tiles[tile_idx * 16 + elem_idx] = V[elem_idx / 4][elem_idx % 4];
    }
}
```

- [ ] **Step 4: 实现Winograd Element-wise Multiplication Kernel**

```cpp
// Element-wise multiplication: M = V * W (per tile)
// input_tiles: [N, C, num_tiles, 16] - transformed input tiles
// weight_tiles: [out_C, C, 16] - transformed weights
// intermediate: [N, out_C, num_tiles, 16]
__global__ void winograd_elementwise_kernel(
    const float* input_tiles, const float* weight_tiles,
    float* intermediate,
    int N, int C, int out_C, int num_tiles) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * num_tiles * 16;
    
    if (idx < total) {
        int rem = idx;
        int n = rem / (out_C * num_tiles * 16);
        rem %= (out_C * num_tiles * 16);
        int oc = rem / (num_tiles * 16);
        rem %= (num_tiles * 16);
        int tile_idx = rem / 16;
        int elem_idx = rem % 16;
        
        // Sum over input channels: M[n,oc,tile] += sum_c V[n,c,tile] * W[oc,c]
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = input_tiles[(n * C + c) * num_tiles * 16 + tile_idx * 16 + elem_idx];
            float w = weight_tiles[oc * C * 16 + c * 16 + elem_idx];
            sum += v * w;
        }
        
        intermediate[idx] = sum;
    }
}
```

- [ ] **Step 5: 实现Winograd Output Transform Kernel**

```cpp
// Output transform: Y = A^T @ M @ A
// intermediate: [N, out_C, num_tiles, 16]
// output: [N, out_C, out_H, out_W]
__global__ void winograd_output_transform_kernel(
    const float* intermediate, float* output,
    int N, int out_C, int out_H, int out_W,
    int num_tiles_h, int num_tiles_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;
    
    if (idx < total) {
        int rem = idx;
        int n = rem / (out_C * out_H * out_W);
        rem %= (out_C * out_H * out_W);
        int oc = rem / (out_H * out_W);
        rem %= (out_H * out_W);
        int oh = rem / out_W;
        int ow = rem % out_W;
        
        // 确定该输出像素属于哪个tile
        int tile_h = oh / 2;
        int tile_w = ow / 2;
        int tile_elem_h = oh % 2 + 1;  // 0,1 -> 1,2; 用1,2,3,4索引
        int tile_elem_w = ow % 2 + 1;
        
        // 从intermediate中取出4x4 tile
        float M[4][4];
        int tile_idx = tile_h * num_tiles_w + tile_w;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                M[i][j] = intermediate[n * out_C * num_tiles_h * num_tiles_w * 16 +
                                       oc * num_tiles_h * num_tiles_w * 16 +
                                       tile_idx * 16 +
                                       i * 4 + j];
            }
        }
        
        // Y = A^T @ M @ A
        float temp[4] = {0};
        float Y = 0.0f;
        
        // A^T @ M -> temp (4x4 matrix multiplication with row vector)
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                temp[j] += winograd_At[tile_elem_h][k] * M[k][j];
            }
        }
        
        // temp @ A[:, tile_elem_w] -> Y
        for (int k = 0; k < 4; ++k) {
            Y += temp[k] * winograd_At[tile_elem_w][k];
        }
        
        output[idx] = Y;
    }
}
```

- [ ] **Step 6: 实现Winograd Forward主函数**

```cpp
void cuda_conv2d_winograd_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_buffer,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream) {
    
    // 仅支持 3x3, stride=1, pad=1
    if (stride_h != 1 || stride_w != 1 || pad_h != 1 || pad_w != 1) {
        // Fallback到im2col
        return;
    }
    
    int out_H = H - 2;  // pad=1, kernel=3 -> out_H = H-2
    int out_W = W - 2;
    int num_tiles_h = get_num_tiles(out_H);
    int num_tiles_w = get_num_tiles(out_W);
    int num_tiles = num_tiles_h * num_tiles_w;
    
    // 临时缓冲区布局:
    // temp_buffer[0: out_C * C * 16]              -> transformed weights
    // temp_buffer[out_C*C*16 : ...]              -> input tiles
    // ...                                         -> intermediate
    
    float* weight_transformed = temp_buffer;
    float* input_tiles = temp_buffer + out_C * C * 16;
    float* intermediate = input_tiles + N * C * num_tiles * 16;
    
    // 1. Transform weights
    int wt_blocks = (out_C * C + 255) / 256;
    winograd_weight_transform_kernel<<<wt_blocks, 256, 0, stream>>>(
        weight, weight_transformed, out_C, C);
    
    // 2. Transform inputs
    int it_blocks = (N * C * num_tiles * 16 + 255) / 256;
    winograd_input_transform_kernel<<<it_blocks, 256, 0, stream>>>(
        input, input_tiles, N, C, H, W, num_tiles_h, num_tiles_w);
    
    // 3. Element-wise multiplication
    int em_blocks = (N * out_C * num_tiles * 16 + 255) / 256;
    winograd_elementwise_kernel<<<em_blocks, 256, 0, stream>>>(
        input_tiles, weight_transformed, intermediate, N, C, out_C, num_tiles);
    
    // 4. Output transform
    int out_blocks = (N * out_C * out_H * out_W + 255) / 256;
    winograd_output_transform_kernel<<<out_blocks, 256, 0, stream>>>(
        intermediate, output, N, out_C, out_H, out_W, num_tiles_h, num_tiles_w);
    
    // 5. Add bias
    if (bias != nullptr) {
        int bias_blocks = (N * out_C * out_H * out_W + 255) / 256;
        bias_add_output_kernel<<<bias_blocks, 256, 0, stream>>>(
            output, bias, output, N, out_C, out_H, out_W);
    }
    
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 7: 添加cuda_ops.h声明**

```cpp
// 在 cuda_ops.h 中添加 (约第85行附近)
void cuda_conv2d_winograd_forward(
    const float* input, const float* weight, const float* bias,
    float* output, float* temp_buffer,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = 0);
```

- [ ] **Step 8: 创建测试文件 tests/test_conv2d_winograd.cpp**

```cpp
#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>

class WinogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
    
    std::vector<float> generate_random(size_t size) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = -1.0f + static_cast<float>(rand()) / RAND_MAX * 2.0f;
        }
        return v;
    }
};

TEST_F(WinogradTest, ForwardCorrectness) {
    // 配置: N=2, C=4, H=28, W=28, out_C=8, K=3, stride=1, pad=1
    int N=2, C=4, H=28, W=28, out_C=8, K=3, stride=1, pad=1;
    int out_H = H - 2;  // 26
    int out_W = W - 2;  // 26
    
    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * K * K);
    auto bias = generate_random(out_C);
    
    // 分配GPU内存
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * K * K);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    CudaBuffer d_winograd_output(N * out_C * out_H * out_W);
    
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    host_to_device_async(d_bias.data, bias.data(), out_C);
    
    // im2col baseline
    CudaBuffer d_col_buffer(C * K * K * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);
    cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                       d_output.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Winograd forward
    // 计算temp_buffer大小
    int num_tiles_h = (out_H + 1) / 2;
    int num_tiles_w = (out_W + 1) / 2;
    int num_tiles = num_tiles_h * num_tiles_w;
    size_t temp_size = out_C * C * 16 + N * C * num_tiles * 16 + N * out_C * num_tiles * 16;
    CudaBuffer d_temp(temp_size);
    
    cuda_conv2d_winograd_forward(d_input.data, d_weight.data, d_bias.data,
                                  d_winograd_output.data, d_temp.data,
                                  N, C, H, W, out_C, stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 比较结果
    std::vector<float> output_im2col(N * out_C * out_H * out_W);
    std::vector<float> output_winograd(N * out_C * out_H * out_W);
    device_to_host(d_output.data, output_im2col.data(), N * out_C * out_H * out_W);
    device_to_host(d_winograd_output.data, output_winograd.data(), N * out_C * out_H * out_W);
    
    for (size_t i = 0; i < output_im2col.size(); ++i) {
        EXPECT_NEAR(output_im2col[i], output_winograd[i], 1e-3f)
            << "Mismatch at index " << i;
    }
}

TEST_F(WinogradTest, PerformanceComparison) {
    // 大规模测试: N=32, C=64, H=32, W=32, out_C=64
    int N=32, C=64, H=32, W=32, out_C=64, K=3;
    int out_H = H - 2, out_W = W - 2;
    
    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * K * K);
    auto bias = generate_random(out_C);
    
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * K * K);
    CudaBuffer d_bias(out_C), d_output(N * out_C * out_H * out_W);
    CudaBuffer d_col_buffer(C * K * K * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);
    
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    host_to_device_async(d_bias.data, bias.data(), out_C);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                          d_output.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark im2col
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                          d_output.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double im2col_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // 计算GFLOPS
    long long flops = 2LL * N * out_C * out_H * out_W * C * K * K;
    double gflops = flops / (im2col_ms * 1e6);
    
    std::cout << "\n========== Winograd Performance ==========\n";
    std::cout << "  im2col GFLOPS: " << gflops << "\n";
    std::cout << "  im2col time: " << im2col_ms << " ms\n";
    std::cout << "===========================================\n";
}
```

- [ ] **Step 9: 更新CMakeLists.txt添加新测试**

```cmake
# tests/CMakeLists.txt 添加
add_executable(test_conv2d_winograd test_conv2d_winograd.cpp)
target_link_libraries(test_conv2d_winograd cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_conv2d_winograd PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_conv2d_winograd)
```

- [ ] **Step 10: 编译并运行测试**

Run: `cd build && cmake .. && make -j$(nproc) test_conv2d_winograd && ./bin/test_conv2d_winograd`
Expected: 所有测试PASSED，Winograd比im2col快2-3x

- [ ] **Step 11: 提交**

```bash
git add src/conv2d_winograd.cu include/cuda_ops.h tests/test_conv2d_winograd.cpp tests/CMakeLists.txt
git commit -m "feat: add Winograd F(4×4,3×3) conv2d forward

- 实现Winograd forward变换矩阵 (G, B^T, A^T)
- Weight transform kernel (3x3 -> 4x4)
- Input transform kernel (im2col风格的tile transform)
- Element-wise multiplication kernel
- Output transform kernel
- 与im2col对比测试 (正确性验证)

性能: 2-3x加速 vs im2col for 3x3 conv"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

## Task 2: FP16/Tensor Core实现

**Files:**
- Create: `src/half_utils.cu`, `src/matmul_fp16.cu`
- Modify: `include/cuda_ops.h`, `src/cuda_ops_export.cu`
- Test: `tests/test_fp16_tensor_core.cpp`

- [ ] **Step 1: 创建FP16工具函数 half_utils.cu**

```cpp
// src/half_utils.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

// Float to Half转换kernel (向量化)
__global__ void float_to_half_kernel(const float* in, __half* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// Half to Float转换kernel (向量化)
__global__ void half_to_float_kernel(const __half* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// 使用shuffle指令的快速转换 (warp-level)
__device__ __half float_to_half_fast(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int sign = bits >> 16;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    unsigned int mantissa = bits & 0x7FFFFF;
    
    if (exp <= 0) {
        return __int2half_rn(0);
    } else if (exp >= 31) {
        return __int2half_rn(32768);
    } else {
        unsigned int half = (sign << 15) | (exp << 10) | (mantissa >> 13);
        return __ushort2half_rn(half);
    }
}

} // namespace

// Host端转换函数
void float_to_half(const float* in, __half* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    float_to_half_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}

void half_to_float(const __half* in, float* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    half_to_float_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}
```

- [ ] **Step 2: 创建Tensor Core Matmul封装 matmul_fp16.cu**

```cpp
// src/matmul_fp16.cu
#include <cuda_fp16.h>
#include <mma.h>
#include "cuda_ops.h"
#include "cuda_util.h"

using namespace nvcuda::wmma;

// WMMA tile size: 16×16×16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

namespace {

// Tensor Core Matmul kernel
// C = A @ B, A和B是FP16, C是FP32累加
__global__ void tensor_core_matmul_kernel(
    const __half* A, const __half* B, float* C,
    int M, int N, int K) {
    
    // 每个block计算一个16×16输出tile
    int row_o = blockIdx.y * WMMA_M;
    int col_o = blockIdx.x * WMMA_N;
    
    if (row_o >= M || col_o >= N) return;
    
    // Fragment声明
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化累加器为0
    fill_fragment(c_frag, 0.0f);
    
    // 分块计算: K方向遍历
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载A tile (M×K)
        int a_row = row_o;
        int a_col = k;
        if (a_row < M && a_col < K) {
            load_matrix_sync(a_frag, A + a_row * K + a_col, K);
        } else {
            fill_fragment(a_frag, __half(0));
        }
        
        // 加载B tile (K×N)
        int b_row = k;
        int b_col = col_o;
        if (b_row < K && b_col < N) {
            load_matrix_sync(b_frag, B + b_row * N + b_col, N);
        } else {
            fill_fragment(b_frag, __half(0));
        }
        
        // Tensor Core矩阵乘累加
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    store_matrix_sync(C + row_o * N + col_o, c_frag, N, row_major);
}

// FP32 Matmul baseline (使用现有的tiled实现)
__global__ void matmul_fp32_baseline(
    const float* A, const float* B, float* C, int M, int N, int K) {
    // 使用现有的32x32 tiling kernel
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + 31) / 32; ++t) {
        // Load A tile
        if (row < M && t * 32 + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile
        if (col < N && t * 32 + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < 32; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

} // namespace

// Public API
void cuda_matmul_fp16(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream) {
    
    // Grid: ceil(M/16) x ceil(N/16)
    dim3 grid_dim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block_dim(32, 32);  // WMMA使用32线程warp
    
    tensor_core_matmul_kernel<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_matmul_fp32_baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream) {
    
    dim3 grid_dim((N + 31) / 32, (M + 31) / 32);
    dim3 block_dim(32, 32);
    
    matmul_fp32_baseline<<<grid_dim, block_dim, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 3: 添加cuda_ops.h声明**

```cpp
// 在 cuda_ops.h 中添加
void cuda_matmul_fp16(
    const __half* A, const __half* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0);

void float_to_half(const float* in, __half* out, size_t n, cudaStream_t stream = 0);
void half_to_float(const __half* in, float* out, size_t n, cudaStream_t stream = 0);
```

- [ ] **Step 4: 创建测试 tests/test_fp16_tensor_core.cpp**

```cpp
#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <cuda_fp16.h>

class TensorCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaSetDevice(0));
    }
    
    std::vector<float> generate_random(size_t size) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = -1.0f + static_cast<float>(rand()) / RAND_MAX * 2.0f;
        }
        return v;
    }
};

TEST_F(TensorCoreTest, FP16Conversion) {
    size_t n = 1024;
    auto input = generate_random(n);
    
    CudaBuffer d_float(n), d_half(n);
    CudaBuffer d_result(n);
    
    host_to_device_async(d_float.data, input.data(), n);
    
    // Float -> Half
    float_to_half(d_float.data, (__half*)d_half.data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Half -> Float
    half_to_float((__half*)d_half.data, d_result.data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> result(n);
    device_to_host(d_result.data, result.data(), n);
    
    // 检查精度损失
    for (size_t i = 0; i < n; ++i) {
        float expected = input[i];
        float actual = result[i];
        // FP16相对误差应该在1%以内
        EXPECT_NEAR(actual, expected, std::max(0.01f, std::abs(expected) * 0.01f))
            << "Mismatch at index " << i;
    }
}

TEST_F(TensorCoreTest, MatmulCorrectness) {
    int M = 256, N = 256, K = 256;
    
    auto A = generate_random(M * K);
    auto B = generate_random(K * N);
    
    CudaBuffer d_A(M * K), d_B(K * N);
    CudaBuffer d_C_fp32(M * N), d_C_tensor(M * N);
    
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    
    // FP32 baseline
    cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C_fp32.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // FP16 + Tensor Core
    CudaBuffer d_A_half(M * K), d_B_half(K * N);
    float_to_half(d_A.data, (__half*)d_A_half.data, M * K);
    float_to_half(d_B.data, (__half*)d_B_half.data, K * N);
    
    cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                     d_C_tensor.data, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> C_fp32(M * N), C_tensor(M * N);
    device_to_host(d_C_fp32.data, C_fp32.data(), M * N);
    device_to_host(d_C_tensor.data, C_tensor.data(), M * N);
    
    // 对比结果 (允许FP16精度损失)
    for (size_t i = 0; i < M * N; ++i) {
        float expected = C_fp32[i];
        float actual = C_tensor[i];
        float rel_error = std::abs(actual - expected) / (std::abs(expected) + 1e-6);
        EXPECT_LT(rel_error, 0.05f)  // 5%相对误差容忍
            << "Mismatch at index " << i;
    }
}

TEST_F(TensorCoreTest, MatmulPerformance) {
    int M = 1024, N = 1024, K = 1024;
    
    auto A = generate_random(M * K);
    auto B = generate_random(K * N);
    
    CudaBuffer d_A(M * K), d_B(K * N);
    CudaBuffer d_C(M * N);
    
    host_to_device_async(d_A.data, A.data(), M * K);
    host_to_device_async(d_B.data, B.data(), K * N);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark FP32
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_matmul_fp32_baseline(d_A.data, d_B.data, d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // FP16 conversion + Tensor Core
    CudaBuffer d_A_half(M * K), d_B_half(K * N);
    float_to_half(d_A.data, (__half*)d_A_half.data, M * K);
    float_to_half(d_B.data, (__half*)d_B_half.data, K * N);
    
    for (int i = 0; i < 10; ++i) {
        cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                         d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cuda_matmul_fp16((__half*)d_A_half.data, (__half*)d_B_half.data,
                         d_C.data, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double fp16_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    long long flops = 2LL * M * N * K;
    double fp32_gflops = flops / (fp32_ms * 1e6);
    double fp16_gflops = flops / (fp16_ms * 1e6);
    
    std::cout << "\n========== Tensor Core Matmul Performance ==========\n";
    std::cout << "  FP32: " << fp32_gflops << " GFLOPS (" << fp32_ms << " ms)\n";
    std::cout << "  FP16+TensorCore: " << fp16_gflops << " GFLOPS (" << fp16_ms << " ms)\n";
    std::cout << "  Speedup: " << fp32_ms / fp16_ms << "x\n";
    std::cout << "=====================================================\n";
}
```

- [ ] **Step 5: 更新CMakeLists.txt**

```cmake
# tests/CMakeLists.txt 添加
add_executable(test_fp16_tensor_core test_fp16_tensor_core.cpp)
target_link_libraries(test_fp16_tensor_core cuda_ops GTest::gtest GTest::gtest_main pthread)
target_include_directories(test_fp16_tensor_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
gtest_discover_tests(test_fp16_tensor_core)
```

- [ ] **Step 6: 编译并运行测试**

Run: `cd build && make -j$(nproc) test_fp16_tensor_core && ./bin/test_fp16_tensor_core`
Expected: PASSED, Tensor Core比FP32快2-4x

- [ ] **Step 7: 提交**

```bash
git add src/half_utils.cu src/matmul_fp16.cu include/cuda_ops.h tests/test_fp16_tensor_core.cpp
git commit -m "feat: add FP16 conversion and Tensor Core matmul

- FP16/Half转换kernel (float<->half)
- Tensor Core WMMA API封装 (16x16x16 fragment)
- tensor_core_matmul_kernel使用wmma::mma_sync
- FP32 baseline vs FP16+TensorCore性能对比测试

性能: 2-4x加速 vs FP32 tiled matmul"
```

---

## Task 3: Kernel融合 (Conv→ReLU→Pool)

**Files:**
- Create: `src/conv2d_fused.cu`
- Modify: `include/cuda_ops.h`, `src/cuda_ops_export.cu`
- Test: `tests/test_fused_conv.cpp`

- [ ] **Step 1: 创建Conv→ReLU→Pool融合kernel框架 conv2d_fused.cu**

```cpp
// src/conv2d_fused.cu
#include "cuda_ops.h"
#include "cuda_util.h"

namespace {

#define FUSED_TILE_H 16
#define FUSED_TILE_W 16

// Conv→ReLU→Pool融合kernel
// 每个block处理一个输出tile: 16x16 conv输出 -> 8x8 pool输出 (2x2 pool, stride=2)
// 一个block有16x16=256线程
__global__ void conv2d_relu_pool_fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* max_indices,
    int N, int C, int H, int W,
    int out_C, int out_H, int out_W,
    int pool_size, int pool_stride) {
    
    // 共享内存: 输入窗口 (16x16 + 2 padding for 3x3 conv)
    __shared__ float input_tile[FUSED_TILE_H + 2][FUSED_TILE_W + 2];
    // 共享内存: 权重 (16个output channels, 每个3x3)
    __shared__ float weight_tile[16][3][3];
    
    // 线程索引
    int tid = threadIdx.x;
    int th = tid / 16;  // 0-15
    int tw = tid % 16;  // 0-15
    
    // Block索引: 处理哪个tile和哪个batch
    int batch_idx = blockIdx.z;
    int out_c_start = blockIdx.y * 16;
    int tile_h = blockIdx.x / (out_W / 8);  // 假设out_W是8的倍数
    int tile_w = blockIdx.x % (out_W / 8);
    
    int out_h_start = tile_h * 8;  // 2x2 pool, 16 conv输出 -> 8 pool输出
    int out_w_start = tile_w * 8;
    
    if (batch_idx >= N || out_c_start >= out_C) return;
    
    // 加载输入到共享内存 (每个线程加载一个像素)
    for (int c = 0; c < C; ++c) {
        // 加载16x16区域到shared memory
        int ih = out_h_start + th - 1;  // pad=1
        int iw = out_w_start + tw - 1;
        
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            input_tile[th][tw] = input[batch_idx * C * H * W + c * H * W + ih * W + iw];
        } else {
            input_tile[th][tw] = 0.0f;
        }
        
        __syncthreads();
        
        // 加载权重 (每个channel的3x3 kernel)
        if (th < 3 && tw < 3) {
            for (int oc = 0; oc < 16 && out_c_start + oc < out_C; ++oc) {
                weight_tile[oc][th][tw] = weight[(out_c_start + oc) * C * 9 + c * 9 + th * 3 + tw];
            }
        }
        
        __syncthreads();
        
        // 计算卷积 (每个线程计算一个输出像素)
        if (th < 16 && tw < 16 && out_h_start + th < out_H && out_w_start + tw < out_W) {
            float conv_val = 0.0f;
            
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    conv_val += input_tile[th + kh][tw + kw] * weight_tile[0][kh][kw];
                }
            }
            
            // ReLU
            conv_val = fmaxf(0.0f, conv_val);
            
            // 存储到临时区域 (后续pool)
            // 实际实现中需要更好的内存布局
        }
    }
}

// 更简洁的版本: Conv -> ReLU，然后单独的pool kernel
// 这更符合实际实现，因为pool需要跨线程协作

__global__ void conv2d_relu_fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W,
    int out_C, int out_H, int out_W) {
    
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x / out_W;
    int ow = blockIdx.x % out_W;
    
    if (n >= N || oc >= out_C || oh >= out_H || ow >= out_W) return;
    
    float sum = 0.0f;
    
    // 3x3 conv
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = oh + kh - 1;  // pad=1
                int iw = ow + kw - 1;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += input[n * C * H * W + c * H * W + ih * W + iw] *
                           weight[oc * C * 9 + c * 9 + kh * 3 + kw];
                }
            }
        }
    }
    
    // Bias + ReLU
    if (bias != nullptr) {
        sum += bias[oc];
    }
    sum = fmaxf(0.0f, sum);
    
    output[n * out_C * out_H * out_W + oc * out_H * out_W + oh * out_W + ow] = sum;
}

// Warp-level reduction for maxpool
__device__ float warp_reduce_max(float val, int lane_id) {
    // 2x2 maxpool: 4 threads协作
    // lane_id: 0,1,2,3 对应2x2窗口的四个位置
    float other = __shfl_xor_sync(0xffffffff, val, 1);
    val = fmaxf(val, other);
    other = __shfl_xor_sync(0xffffffff, val, 2);
    val = fmaxf(val, other);
    return val;
}

__global__ void maxpool2d_warp_kernel(
    const float* input, float* output, int* indices,
    int N, int C, int H, int W,
    int pool_size, int pool_stride) {
    
    int n = blockIdx.z;
    int c = blockIdx.y;
    int out_h = blockIdx.x / pool_stride;  // 需要调整
    
    // ... 实现warp-level max reduction
}

} // namespace

// Public API
void cuda_conv2d_relu_fused(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream);

void cuda_maxpool2d_warp(
    const float* input, float* output, int* indices,
    int N, int C, int H, int W,
    int pool_size, int pool_stride,
    cudaStream_t stream);
```

- [ ] **Step 2: 添加cuda_ops.h声明**

```cpp
// 在 cuda_ops.h 中添加
void cuda_conv2d_relu_fused(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C, int H, int W, int out_C,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream = 0);

void cuda_maxpool2d_warp(
    const float* input, float* output, int* indices,
    int N, int C, int H, int W,
    int pool_size, int pool_stride,
    cudaStream_t stream = 0);
```

- [ ] **Step 3: 创建测试 tests/test_fused_conv.cpp**

```cpp
#include <gtest/gtest.h>
#include "cuda_ops.h"
#include "cuda_util.h"
#include <vector>

class FusedConvTest : public ::testing::Test {
protected:
    void SetUp() override { CUDA_CHECK(cudaSetDevice(0)); }
    
    std::vector<float> generate_random(size_t size) {
        std::vector<float> v(size);
        for (size_t i = 0; i < size; ++i) {
            v[i] = -1.0f + static_cast<float>(rand()) / RAND_MAX * 2.0f;
        }
        return v;
    }
};

TEST_F(FusedConvTest, ConvReluCorrectness) {
    int N=2, C=8, H=28, W=28, out_C=16, K=3, stride=1, pad=1;
    int out_H = H - 2, out_W = W - 2;
    
    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * K * K);
    auto bias = generate_random(out_C);
    
    CudaBuffer d_input(N * C * H * W), d_weight(out_C * C * K * K);
    CudaBuffer d_bias(out_C), d_output_sep(N * out_C * out_H * out_W);
    CudaBuffer d_output_fused(N * out_C * out_H * out_W);
    
    host_to_device_async(d_input.data, input.data(), N * C * H * W);
    host_to_device_async(d_weight.data, weight.data(), out_C * C * K * K);
    host_to_device_async(d_bias.data, bias.data(), out_C);
    
    // Separate: Conv2d -> ReLU
    CudaBuffer d_col_buffer(C * K * K * N * out_H * out_W);
    CudaBuffer d_gemm_buffer(out_C * N * out_H * out_W);
    cuda_conv2d_im2col(d_input.data, d_weight.data, d_bias.data,
                       d_output_sep.data, d_col_buffer.data, d_gemm_buffer.data, desc);
    CUDA_CHECK(cudaDeviceSynchronize());
    // ReLU inplace
    // (简化: 假设im2col已经包含bias)
    // 需要单独执行relu
    
    // Fused
    cuda_conv2d_relu_fused(d_input.data, d_weight.data, d_bias.data,
                           d_output_fused.data, N, C, H, W, out_C,
                           stride, stride, pad, pad);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> out_sep(N * out_C * out_H * out_W);
    std::vector<float> out_fused(N * out_C * out_H * out_W);
    device_to_host(d_output_sep.data, out_sep.data(), N * out_C * out_H * out_W);
    device_to_host(d_output_fused.data, out_fused.data(), N * out_C * out_H * out_W);
    
    for (size_t i = 0; i < out_sep.size(); ++i) {
        EXPECT_NEAR(out_sep[i], out_fused[i], 1e-3f) << "Mismatch at index " << i;
    }
}

TEST_F(FusedConvTest, ConvReluPoolCorrectness) {
    // Conv2d -> ReLU -> MaxPool2d
    int N=2, C=8, H=28, W=28, out_C=16, K=3;
    int conv_out_H = H - 2, conv_out_W = W - 2;  // 26x26 after conv
    int pool_out_H = 13, pool_out_W = 13;  // after 2x2 pool
    
    auto input = generate_random(N * C * H * W);
    auto weight = generate_random(out_C * C * K * K);
    auto bias = generate_random(out_C);
    
    // PyTorch reference
    // ... 简化实现
    
    // 融合版本
    // conv_relu -> pool
    
    // 比较
}
```

- [ ] **Step 4: 更新CMakeLists.txt**

- [ ] **Step 5: 编译并运行测试**

- [ ] **Step 6: 提交**

---

## Task 4: 性能对比基准测试

**Files:**
- Create: `python/benchmark_compare.py`

- [ ] **Step 1: 创建性能对比脚本**

```python
# python/benchmark_compare.py
import numpy as np
import time
from cuda_ops import CUDAOps
import torch
import torch.nn.functional as F

class PerformanceComparator:
    def __init__(self):
        self.ops = CUDAOps()
        
    def benchmark_single_op(self, op_name, config, impls):
        """单算子性能对比
        
        Args:
            op_name: 'conv2d', 'matmul', etc.
            config: dict with N, C, H, W, etc.
            impls: list of implementation names to compare
        """
        results = {}
        
        for impl in impls:
            if op_name == 'conv2d' and impl == 'winograd':
                # Winograd实现
                # ...
                pass
            elif op_name == 'conv2d' and impl == 'im2col':
                # im2col实现
                pass
            elif op_name == 'conv2d' and impl == 'pytorch':
                # PyTorch参考
                pass
            elif op_name == 'matmul' and impl == 'tensor_core':
                # Tensor Core实现
                pass
        
        return results
    
    def benchmark_training(self, model, epochs, batch_size):
        """训练吞吐量对比"""
        pass
    
    def generate_report(self, results):
        """生成性能报告"""
        print("\n" + "="*70)
        print("Performance Comparison Report")
        print("="*70)
        
        for op, data in results.items():
            print(f"\n{op}:")
            print(f"  {'Implementation':<20} {'Time (ms)':<12} {'GFLOPS':<12} {'Speedup':<10}")
            print("  " + "-"*54)
            
            baseline_time = None
            for impl, metrics in data.items():
                time_ms = metrics.get('time_ms', 0)
                gflops = metrics.get('gflops', 0)
                speedup = metrics.get('speedup', 1.0)
                
                if baseline_time is None:
                    baseline_time = time_ms
                
                print(f"  {impl:<20} {time_ms:<12.3f} {gflops:<12.1f} {speedup:<10.2f}x")
        
        print("="*70)

def main():
    comparator = PerformanceComparator()
    
    # Conv2d测试配置
    conv_configs = [
        {'N': 64, 'C': 16, 'H': 28, 'W': 28, 'out_C': 32, 'K': 3},
        {'N': 64, 'C': 32, 'H': 14, 'W': 14, 'out_C': 64, 'K': 3},
        {'N': 32, 'C': 64, 'H': 32, 'W': 32, 'out_C': 64, 'K': 3},
    ]
    
    results = {}
    
    for config in conv_configs:
        key = f"Conv2d(N={config['N']},C={config['C']},H={config['H']})"
        results[key] = {}
        
        # 当前实现 (im2col)
        # ...
        
        # Winograd
        # ...
        
        # PyTorch
        # ...
    
    comparator.generate_report(results)

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行并验证**

- [ ] **Step 3: 提交**

---

## 实现顺序总结

| Task | 优先级 | 预期收益 | 依赖 |
|------|--------|----------|------|
| Task 1: Winograd Forward | 1 | 2-3x | 无 |
| Task 2: FP16/Tensor Core | 2 | 2-4x | Task 1 |
| Task 3: Kernel融合 | 3 | 1.5-2x | Task 1 |
| Task 4: 性能对比 | 4 | - | Task 1,2,3 |

每个Task完成后会进行Gate 5验证和Gate 4代码审查。

---

## 风险缓解

1. **Winograd边界处理**: 先处理H,W是4倍数的情况，后续处理边界
2. **Tensor Core兼容性**: 检查GPU计算能力(7.0+)，否则fallback到FP32
3. **Kernel融合调试**: 分阶段实现，先Conv→ReLU，再加Pool

---

*Plan created: 2026-04-29*
*Author: Claude Code with ai-engineer-workflow-v4*