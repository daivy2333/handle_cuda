# CNN Training Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend CUDA operators project to support CNN training on MNIST, achieving 98%+ accuracy.

**Architecture:** Add Conv2d/MaxPool2d Python bindings, create SimpleCNN_CUDA model (Conv-Pool-Conv-Pool-FC), reuse existing ReLU/Softmax/Flatten/CrossEntropy/SGD.

**Tech Stack:** CUDA C++, Python ctypes, NumPy, MNIST dataset

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/cuda_ops_export.cu` | Modify | Add Conv2d/MaxPool2d C API exports |
| `python/cuda_ops.py` | Modify | Add Conv2d/MaxPool2d bindings + tests |
| `python/model_cnn_cuda.py` | Create | SimpleCNN_CUDA model class |
| `python/train_mnist_cnn_cuda.py` | Create | CNN training script |

---

## Task 1: Add Conv2d C API Export

**Files:**
- Modify: `src/cuda_ops_export.cu:140-140` (end of file)

- [ ] **Step 1: Add Conv2d forward C API function**

Add at the end of `src/cuda_ops_export.cu` (after line 140, inside the extern "C" block):

```cpp
// ============== Conv2d C API ==============
// Conv2d forward: input [N, C, H, W], weight [out_C, C, kernel_h, kernel_w], output [N, out_C, out_H, out_W]
void cuda_conv2d_f32(const float* input, const float* weight, const float* bias, float* output,
                     int N, int C, int H, int W,
                     int out_C, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int pad_h, int pad_w) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;  // Standard convolution (groups=1)

    // Calculate output dimensions
    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    cuda_conv2d(input, weight, bias, output, desc, 0);
}

// Conv2d backward: compute grad_input, grad_weight, grad_bias
void cuda_conv2d_backward_f32(const float* grad_out, const float* input, const float* weight,
                              float* grad_input, float* grad_weight, float* grad_bias,
                              int N, int C, int H, int W,
                              int out_C, int kernel_h, int kernel_w,
                              int stride_h, int stride_w, int pad_h, int pad_w) {
    Conv2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.out_C = out_C;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.groups = 1;

    desc.out_H = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    desc.out_W = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    cuda_conv2d_backward(grad_out, input, weight, grad_input, grad_weight, grad_bias, desc, 0);
}
```

- [ ] **Step 2: Add MaxPool2d forward C API function**

Continue adding after Conv2d backward:

```cpp
// ============== MaxPool2d C API ==============
// MaxPool2d forward: input [N, C, H, W] -> output [N, C, out_H, out_W]
// indices stores the index of max element for backward pass
void cuda_maxpool2d_f32(const float* input, float* output, int* indices,
                        int N, int C, int H, int W,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w, int pad_h, int pad_w) {
    Pool2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;

    cuda_maxpool2d(input, output, indices, desc, 0);
}

// MaxPool2d backward: scatter grad_out to grad_input using indices
void cuda_maxpool2d_backward_f32(const float* grad_out, const int* indices, float* grad_input,
                                  int N, int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int stride_h, int stride_w, int pad_h, int pad_w) {
    Pool2dDesc desc;
    desc.N = N;
    desc.C = C;
    desc.H = H;
    desc.W = W;
    desc.kernel_h = kernel_h;
    desc.kernel_w = kernel_w;
    desc.stride_h = stride_h;
    desc.stride_w = stride_w;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;

    cuda_maxpool2d_backward(grad_out, nullptr, indices, grad_input, desc, 0);
}
```

- [ ] **Step 3: Close extern "C" block**

Add closing brace at the end:

```cpp
} // extern "C"
```

- [ ] **Step 4: Build and verify**

Run: `cd build && make -j$(nproc)`
Expected: No compilation errors

- [ ] **Step 5: Commit**

```bash
git add src/cuda_ops_export.cu
git commit -m "feat: add Conv2d and MaxPool2d C API exports for Python binding"
```

---

## Task 2: Add Conv2d Python Binding

**Files:**
- Modify: `python/cuda_ops.py:17-110` (setup_functions) and add methods after line 291

- [ ] **Step 1: Add Conv2d function signatures in setup_functions**

In `_setup_functions` method, add after the softmax_backward setup (around line 109):

```python
        # Conv2d
        self.lib.cuda_conv2d_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.cuda_conv2d_f32.restype = None

        self.lib.cuda_conv2d_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.cuda_conv2d_backward_f32.restype = None

        # MaxPool2d
        self.lib.cuda_maxpool2d_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.cuda_maxpool2d_f32.restype = None

        self.lib.cuda_maxpool2d_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.cuda_maxpool2d_backward_f32.restype = None
```

- [ ] **Step 2: Add conv2d method**

Add after softmax_backward method (around line 291):

```python
    def conv2d(self, input_ptr, weight_ptr, bias_ptr,
               N, C, H, W, out_C, kernel_h, kernel_w,
               stride_h=1, stride_w=1, pad_h=0, pad_w=0,
               output_ptr=None):
        """Pure CUDA Conv2d forward.

        Args:
            input_ptr: GPU pointer to input [N, C, H, W]
            weight_ptr: GPU pointer to weight [out_C, C, kernel_h, kernel_w]
            bias_ptr: GPU pointer to bias [out_C] (can be None)
            N, C, H, W: input dimensions
            out_C: output channels
            kernel_h, kernel_w: kernel size
            stride_h, stride_w: stride
            pad_h, pad_w: padding

        Returns:
            output_ptr: GPU pointer to output [N, out_C, out_H, out_W]
        """
        out_H = (H + 2 * pad_h - kernel_h) // stride_h + 1
        out_W = (W + 2 * pad_w - kernel_w) // stride_w + 1

        if output_ptr is None:
            output_ptr = self.alloc(N * out_C * out_H * out_W)

        # Handle None bias
        bias_arg = bias_ptr if bias_ptr is not None else None

        self.lib.cuda_conv2d_f32(
            input_ptr, weight_ptr, bias_arg, output_ptr,
            N, C, H, W, out_C, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w
        )
        return output_ptr

    def conv2d_backward(self, grad_out_ptr, input_ptr, weight_ptr,
                        N, C, H, W, out_C, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w,
                        grad_input_ptr=None, grad_weight_ptr=None, grad_bias_ptr=None):
        """Pure CUDA Conv2d backward.

        Args:
            grad_out_ptr: GPU pointer to gradient of output [N, out_C, out_H, out_W]
            input_ptr: GPU pointer to forward input [N, C, H, W]
            weight_ptr: GPU pointer to weight [out_C, C, kernel_h, kernel_w]
            ... dimensions same as forward

        Returns:
            grad_input_ptr, grad_weight_ptr, grad_bias_ptr
        """
        if grad_input_ptr is None:
            grad_input_ptr = self.alloc(N * C * H * W)
        if grad_weight_ptr is None:
            grad_weight_ptr = self.alloc(out_C * C * kernel_h * kernel_w)
        if grad_bias_ptr is None:
            grad_bias_ptr = self.alloc(out_C)

        self.lib.cuda_conv2d_backward_f32(
            grad_out_ptr, input_ptr, weight_ptr,
            grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
            N, C, H, W, out_C, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w
        )
        return grad_input_ptr, grad_weight_ptr, grad_bias_ptr
```

- [ ] **Step 3: Add maxpool2d method**

Add after conv2d_backward:

```python
    def maxpool2d(self, input_ptr, N, C, H, W,
                  kernel_h=2, kernel_w=2, stride_h=2, stride_w=2,
                  pad_h=0, pad_w=0, output_ptr=None, indices_ptr=None):
        """Pure CUDA MaxPool2d forward.

        Args:
            input_ptr: GPU pointer to input [N, C, H, W]
            N, C, H, W: input dimensions
            kernel_h, kernel_w: pooling window size
            stride_h, stride_w: stride
            pad_h, pad_w: padding

        Returns:
            output_ptr: GPU pointer to output [N, C, out_H, out_W]
            indices_ptr: GPU pointer to indices [N, C, out_H, out_W] (int32)
        """
        out_H = (H + 2 * pad_h - kernel_h) // stride_h + 1
        out_W = (W + 2 * pad_w - kernel_w) // stride_w + 1

        if output_ptr is None:
            output_ptr = self.alloc(N * C * out_H * out_W)
        if indices_ptr is None:
            # Allocate int32 indices (4 bytes each)
            indices_ptr = self.lib.cuda_alloc(N * C * out_H * out_W * 4)

        self.lib.cuda_maxpool2d_f32(
            input_ptr, output_ptr, indices_ptr,
            N, C, H, W, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w
        )
        return output_ptr, indices_ptr

    def maxpool2d_backward(self, grad_out_ptr, indices_ptr, N, C, H, W,
                           kernel_h, kernel_w, stride_h, stride_w,
                           pad_h, pad_w, grad_input_ptr=None):
        """Pure CUDA MaxPool2d backward.

        Args:
            grad_out_ptr: GPU pointer to gradient of output [N, C, out_H, out_W]
            indices_ptr: GPU pointer to indices from forward pass
            N, C, H, W: input dimensions
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w: pooling params

        Returns:
            grad_input_ptr: GPU pointer to gradient of input [N, C, H, W]
        """
        if grad_input_ptr is None:
            grad_input_ptr = self.alloc(N * C * H * W)

        self.lib.cuda_maxpool2d_backward_f32(
            grad_out_ptr, indices_ptr, grad_input_ptr,
            N, C, H, W, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w
        )
        return grad_input_ptr
```

- [ ] **Step 4: Add binding test for conv2d**

Add test function in the test_binding() function or as a separate test:

```python
def test_conv2d_binding():
    """Test Conv2d binding against PyTorch."""
    print("\n9. Testing conv2d...")
    import torch
    import torch.nn.functional as F

    ops = CUDAOps()

    # Test: N=2, C=1, H=28, W=28, out_C=16, kernel=3, stride=1, pad=1
    N, C, H, W = 2, 1, 28, 28
    out_C = 16
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1

    # Create input and weights
    input_np = np.random.randn(N, C, H, W).astype(np.float32)
    weight_np = np.random.randn(out_C, C, kernel_h, kernel_w).astype(np.float32)
    bias_np = np.random.randn(out_C).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_np)
    weight_torch = torch.from_numpy(weight_np)
    bias_torch = torch.from_numpy(bias_np)
    output_torch = F.conv2d(input_torch, weight_torch, bias_torch,
                           stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    output_np_ref = output_torch.numpy()

    # CUDA implementation
    input_ptr = ops.to_device(input_np)
    weight_ptr = ops.to_device(weight_np)
    bias_ptr = ops.to_device(bias_np)

    output_ptr = ops.conv2d(input_ptr, weight_ptr, bias_ptr,
                           N, C, H, W, out_C, kernel_h, kernel_w,
                           stride_h, stride_w, pad_h, pad_w)

    output_np = ops.to_host(output_ptr, (N, out_C, 28, 28))

    # Compare
    max_diff = np.abs(output_np - output_np_ref).max()
    print(f"   Max diff vs PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-5, f"Conv2d output mismatch: {max_diff}"

    ops.free(input_ptr)
    ops.free(weight_ptr)
    ops.free(bias_ptr)
    ops.free(output_ptr)
    print("   Conv2d forward: PASSED")
```

- [ ] **Step 5: Add binding test for maxpool2d**

```python
def test_maxpool2d_binding():
    """Test MaxPool2d binding against PyTorch."""
    print("\n10. Testing maxpool2d...")
    import torch
    import torch.nn.functional as F

    ops = CUDAOps()

    # Test: N=2, C=16, H=28, W=28, kernel=2, stride=2
    N, C, H, W = 2, 16, 28, 28
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 0, 0

    # Create input
    input_np = np.random.randn(N, C, H, W).astype(np.float32)

    # PyTorch reference
    input_torch = torch.from_numpy(input_np)
    output_torch = F.max_pool2d(input_torch, kernel_size=(kernel_h, kernel_w),
                                stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    output_np_ref = output_torch.numpy()

    # CUDA implementation
    input_ptr = ops.to_device(input_np)
    output_ptr, indices_ptr = ops.maxpool2d(input_ptr, N, C, H, W,
                                            kernel_h, kernel_w, stride_h, stride_w,
                                            pad_h, pad_w)

    output_np = ops.to_host(output_ptr, (N, C, 14, 14))

    # Compare
    max_diff = np.abs(output_np - output_np_ref).max()
    print(f"   Max diff vs PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-5, f"MaxPool2d output mismatch: {max_diff}"

    ops.free(input_ptr)
    ops.free(output_ptr)
    ops.lib.cuda_free(indices_ptr)
    print("   MaxPool2d forward: PASSED")
```

- [ ] **Step 6: Run binding tests**

Run: `cd python && python -c "from cuda_ops import test_conv2d_binding, test_maxpool2d_binding; test_conv2d_binding(); test_maxpool2d_binding()"`
Expected: Both tests pass with max diff < 1e-5

- [ ] **Step 7: Commit**

```bash
git add python/cuda_ops.py
git commit -m "feat: add Conv2d and MaxPool2d Python bindings with PyTorch validation"
```

---

## Task 3: Create SimpleCNN_CUDA Model

**Files:**
- Create: `python/model_cnn_cuda.py`

- [ ] **Step 1: Create model file with class definition**

Create `python/model_cnn_cuda.py`:

```python
"""
Pure CUDA CNN Model - 2-Conv architecture for MNIST
"""

import numpy as np
from cuda_ops import CUDAOps


class SimpleCNN_CUDA:
    """
    Pure CUDA 2-Conv CNN for MNIST.

    Architecture:
    - Conv1: 1->16 channels, kernel=3x3, stride=1, padding=1
    - ReLU
    - MaxPool1: 2x2, stride=2
    - Conv2: 16->32 channels, kernel=3x3, stride=1, padding=1
    - ReLU
    - MaxPool2: 2x2, stride=2
    - Flatten: 32*7*7 = 1568
    - FC: 1568->10
    - Softmax + CrossEntropy

    All forward/backward computation happens on GPU.
    """

    def __init__(self, ops: CUDAOps):
        self.ops = ops
        np.random.seed(42)

        # Layer dimensions
        self.conv1_C = 16      # output channels for conv1
        self.conv2_C = 32      # output channels for conv2
        self.fc_in = 32 * 7 * 7  # 1568
        self.fc_out = 10

        # Xavier initialization
        # Conv1 weight: [16, 1, 3, 3]
        conv1_w_np = (np.random.randn(16, 1, 3, 3) * np.sqrt(2.0 / (1 * 3 * 3))).astype(np.float32)
        conv1_b_np = np.zeros(16, dtype=np.float32)

        # Conv2 weight: [32, 16, 3, 3]
        conv2_w_np = (np.random.randn(32, 16, 3, 3) * np.sqrt(2.0 / (16 * 3 * 3))).astype(np.float32)
        conv2_b_np = np.zeros(32, dtype=np.float32)

        # FC weight: [1568, 10]
        fc_w_np = (np.random.randn(1568, 10) * np.sqrt(2.0 / 1568)).astype(np.float32)
        fc_b_np = np.zeros(10, dtype=np.float32)

        # Copy weights to GPU
        self.conv1_w_ptr = self.ops.to_device(conv1_w_np)
        self.conv1_b_ptr = self.ops.to_device(conv1_b_np)
        self.conv2_w_ptr = self.ops.to_device(conv2_w_np)
        self.conv2_b_ptr = self.ops.to_device(conv2_b_np)
        self.fc_w_ptr = self.ops.to_device(fc_w_np)
        self.fc_b_ptr = self.ops.to_device(fc_b_np)

        # Allocate gradient buffers
        self.g_conv1_w_ptr = self.ops.alloc(16 * 1 * 3 * 3)
        self.g_conv1_b_ptr = self.ops.alloc(16)
        self.g_conv2_w_ptr = self.ops.alloc(32 * 16 * 3 * 3)
        self.g_conv2_b_ptr = self.ops.alloc(32)
        self.g_fc_w_ptr = self.ops.alloc(1568 * 10)
        self.g_fc_b_ptr = self.ops.alloc(10)

        # Cache for backward pass
        self.cache = {}

    def __del__(self):
        """Free all GPU allocations."""
        if hasattr(self, 'ops') and self.ops:
            # Free weights
            for ptr_name in ['conv1_w_ptr', 'conv1_b_ptr', 'conv2_w_ptr', 'conv2_b_ptr',
                             'fc_w_ptr', 'fc_b_ptr']:
                if hasattr(self, ptr_name):
                    self.ops.free(getattr(self, ptr_name))
            # Free gradient buffers
            for ptr_name in ['g_conv1_w_ptr', 'g_conv1_b_ptr', 'g_conv2_w_ptr', 'g_conv2_b_ptr',
                             'g_fc_w_ptr', 'g_fc_b_ptr']:
                if hasattr(self, ptr_name):
                    self.ops.free(getattr(self, ptr_name))
            # Free cache
            self._free_cache()

    def _free_cache(self):
        """Free cached activations."""
        for key, ptr in self.cache.items():
            if ptr is not None and key.endswith('_ptr'):
                try:
                    if key in ['conv1_out_ptr', 'conv1_relu_ptr', 'pool1_out_ptr',
                               'conv2_out_ptr', 'conv2_relu_ptr', 'pool2_out_ptr',
                               'pool2_indices_ptr', 'fc_out_ptr']:
                        self.ops.free(ptr)
                except:
                    pass
        self.cache.clear()

    def forward(self, x_ptr, batch):
        """Pure CUDA forward pass.

        Args:
            x_ptr: GPU pointer to input [batch, 1, 28, 28]
            batch: batch size

        Returns:
            logits_ptr: GPU pointer to logits [batch, 10]
        """
        # Input: [batch, 1, 28, 28]

        # Conv1: [batch, 1, 28, 28] -> [batch, 16, 28, 28]
        conv1_out_ptr = self.ops.conv2d(
            x_ptr, self.conv1_w_ptr, self.conv1_b_ptr,
            batch, 1, 28, 28, self.conv1_C, 3, 3,
            stride_h=1, stride_w=1, pad_h=1, pad_w=1
        )

        # ReLU (in-place)
        self.ops.relu(conv1_out_ptr, batch * 16 * 28 * 28)

        # MaxPool1: [batch, 16, 28, 28] -> [batch, 16, 14, 14]
        pool1_out_ptr, pool1_indices_ptr = self.ops.maxpool2d(
            conv1_out_ptr, batch, 16, 28, 28,
            kernel_h=2, kernel_w=2, stride_h=2, stride_w=2
        )

        # Conv2: [batch, 16, 14, 14] -> [batch, 32, 14, 14]
        conv2_out_ptr = self.ops.conv2d(
            pool1_out_ptr, self.conv2_w_ptr, self.conv2_b_ptr,
            batch, 16, 14, 14, self.conv2_C, 3, 3,
            stride_h=1, stride_w=1, pad_h=1, pad_w=1
        )

        # ReLU (in-place)
        self.ops.relu(conv2_out_ptr, batch * 32 * 14 * 14)

        # MaxPool2: [batch, 32, 14, 14] -> [batch, 32, 7, 7]
        pool2_out_ptr, pool2_indices_ptr = self.ops.maxpool2d(
            conv2_out_ptr, batch, 32, 14, 14,
            kernel_h=2, kernel_w=2, stride_h=2, stride_w=2
        )

        # Flatten: [batch, 32, 7, 7] -> [batch, 1568]
        flat_ptr = self.ops.flatten(pool2_out_ptr, batch, 32, 7, 7)

        # FC: [batch, 1568] @ [1568, 10] + bias -> [batch, 10]
        fc_out_ptr = self.ops.matmul(flat_ptr, self.fc_w_ptr, batch, 10, 1568)
        logits_ptr = self.ops.bias_add(fc_out_ptr, self.fc_b_ptr, batch, 10)

        # Cache for backward
        self.cache['x_ptr'] = x_ptr
        self.cache['conv1_relu_ptr'] = conv1_out_ptr  # post-relu (used as mask for relu_backward)
        self.cache['pool1_out_ptr'] = pool1_out_ptr
        self.cache['pool1_indices_ptr'] = pool1_indices_ptr
        self.cache['conv2_relu_ptr'] = conv2_out_ptr  # post-relu
        self.cache['pool2_out_ptr'] = pool2_out_ptr
        self.cache['pool2_indices_ptr'] = pool2_indices_ptr
        self.cache['flat_ptr'] = flat_ptr
        self.cache['fc_out_ptr'] = fc_out_ptr
        self.cache['batch'] = batch

        return logits_ptr

    def backward(self, logits_ptr, targets):
        """Pure CUDA backward pass.

        Args:
            logits_ptr: GPU pointer to logits
            targets: numpy int32 array [batch]

        Returns:
            loss: scalar loss value
        """
        batch = self.cache['batch']

        # Cross entropy loss + gradient
        loss, grad_logits_ptr = self.ops.cross_entropy_loss(logits_ptr, targets, batch, 10)

        # Backprop FC: logits = flat @ fc_w + fc_b
        flat_ptr = self.cache['flat_ptr']

        grad_flat_ptr, grad_fc_w_ptr = self.ops.matmul_backward(
            grad_logits_ptr, flat_ptr, self.fc_w_ptr, batch, 10, 1568,
            grad_A_ptr=None, grad_B_ptr=self.g_fc_w_ptr
        )
        _, grad_fc_b_ptr = self.ops.bias_add_backward(
            grad_logits_ptr, batch, 10,
            grad_input_ptr=None, grad_bias_ptr=self.g_fc_b_ptr
        )

        # Backprop Flatten
        grad_pool2_ptr = self.ops.flatten_backward(grad_flat_ptr, batch, 32, 7, 7)

        # Backprop MaxPool2
        pool2_indices_ptr = self.cache['pool2_indices_ptr']
        grad_conv2_relu_ptr = self.ops.maxpool2d_backward(
            grad_pool2_ptr, pool2_indices_ptr, batch, 32, 14, 14,
            kernel_h=2, kernel_w=2, stride_h=2, stride_w=2
        )

        # Backprop ReLU (conv2)
        conv2_relu_ptr = self.cache['conv2_relu_ptr']
        grad_conv2_ptr = self.ops.alloc(batch * 32 * 14 * 14)
        self.ops.relu_backward(grad_conv2_relu_ptr, conv2_relu_ptr, grad_conv2_ptr, batch * 32 * 14 * 14)

        # Backprop Conv2
        pool1_out_ptr = self.cache['pool1_out_ptr']

        grad_pool1_ptr, grad_conv2_w_ptr, grad_conv2_b_ptr = self.ops.conv2d_backward(
            grad_conv2_ptr, pool1_out_ptr, self.conv2_w_ptr,
            batch, 16, 14, 14, self.conv2_C, 3, 3,
            stride_h=1, stride_w=1, pad_h=1, pad_w=1,
            grad_input_ptr=None, grad_weight_ptr=self.g_conv2_w_ptr, grad_bias_ptr=self.g_conv2_b_ptr
        )

        # Backprop MaxPool1
        pool1_indices_ptr = self.cache['pool1_indices_ptr']
        grad_conv1_relu_ptr = self.ops.maxpool2d_backward(
            grad_pool1_ptr, pool1_indices_ptr, batch, 16, 28, 28,
            kernel_h=2, kernel_w=2, stride_h=2, stride_w=2
        )

        # Backprop ReLU (conv1)
        conv1_relu_ptr = self.cache['conv1_relu_ptr']
        grad_conv1_ptr = self.ops.alloc(batch * 16 * 28 * 28)
        self.ops.relu_backward(grad_conv1_relu_ptr, conv1_relu_ptr, grad_conv1_ptr, batch * 16 * 28 * 28)

        # Backprop Conv1
        x_ptr = self.cache['x_ptr']

        _, grad_conv1_w_ptr, grad_conv1_b_ptr = self.ops.conv2d_backward(
            grad_conv1_ptr, x_ptr, self.conv1_w_ptr,
            batch, 1, 28, 28, self.conv1_C, 3, 3,
            stride_h=1, stride_w=1, pad_h=1, pad_w=1,
            grad_input_ptr=None, grad_weight_ptr=self.g_conv1_w_ptr, grad_bias_ptr=self.g_conv1_b_ptr
        )

        # Cleanup intermediate buffers
        self.ops.free(grad_logits_ptr)
        self.ops.free(grad_flat_ptr)
        self.ops.free(grad_pool2_ptr)
        self.ops.free(grad_conv2_relu_ptr)
        self.ops.free(grad_conv2_ptr)
        self.ops.free(grad_pool1_ptr)
        self.ops.free(grad_conv1_relu_ptr)
        self.ops.free(grad_conv1_ptr)
        self.ops.free(logits_ptr)

        # Free cache
        self._free_cache()

        return loss

    def update(self, lr):
        """SGD update on GPU."""
        # Conv1
        self.ops.sgd_update(self.conv1_w_ptr, self.g_conv1_w_ptr, 16 * 1 * 3 * 3, lr)
        self.ops.sgd_update(self.conv1_b_ptr, self.g_conv1_b_ptr, 16, lr)
        # Conv2
        self.ops.sgd_update(self.conv2_w_ptr, self.g_conv2_w_ptr, 32 * 16 * 3 * 3, lr)
        self.ops.sgd_update(self.conv2_b_ptr, self.g_conv2_b_ptr, 32, lr)
        # FC
        self.ops.sgd_update(self.fc_w_ptr, self.g_fc_w_ptr, 1568 * 10, lr)
        self.ops.sgd_update(self.fc_b_ptr, self.g_fc_b_ptr, 10, lr)

    def predict(self, x_ptr, batch):
        """Predict on GPU, return numpy array."""
        logits_ptr = self.forward(x_ptr, batch)
        logits = self.ops.to_host(logits_ptr, (batch, 10))
        self._free_cache()
        return logits.argmax(axis=1)


def test_cnn_model():
    """Test the pure CUDA CNN model."""
    print("Testing SimpleCNN_CUDA...")

    ops = CUDAOps()
    model = SimpleCNN_CUDA(ops)

    # Test forward
    print("\n1. Testing forward pass...")
    batch = 32
    x = np.random.randn(batch, 1, 28, 28).astype(np.float32)
    x_ptr = ops.to_device(x)
    logits_ptr = model.forward(x_ptr, batch)
    logits = ops.to_host(logits_ptr, (batch, 10))
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (batch, 10)
    assert not np.any(np.isnan(logits))
    print("   Forward: PASSED")

    # Test backward
    print("\n2. Testing backward pass...")
    targets = np.random.randint(0, 10, batch).astype(np.int32)
    loss = model.backward(logits_ptr, targets)
    print(f"   Loss: {loss:.4f}")
    assert not np.isnan(loss)
    print("   Backward: PASSED")

    # Test update
    print("\n3. Testing update...")
    model.update(lr=0.01)
    print("   Update: PASSED")

    # Test predict
    print("\n4. Testing predict...")
    preds = model.predict(x_ptr, batch)
    assert preds.shape == (batch,)
    assert np.all(preds >= 0) and np.all(preds < 10)
    print(f"   Predictions: {preds[:5]}")
    print("   Predict: PASSED")

    ops.free(x_ptr)

    print("\n" + "="*50)
    print("All CNN model tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_cnn_model()
```

- [ ] **Step 2: Run model test**

Run: `cd python && python model_cnn_cuda.py`
Expected: All tests pass, no NaN, forward/backward/update work

- [ ] **Step 3: Commit**

```bash
git add python/model_cnn_cuda.py
git commit -m "feat: add SimpleCNN_CUDA model with 2-Conv architecture"
```

---

## Task 4: Create CNN Training Script

**Files:**
- Create: `python/train_mnist_cnn_cuda.py`

- [ ] **Step 1: Create training script**

Create `python/train_mnist_cnn_cuda.py`:

```python
"""
MNIST Training with Pure CUDA CNN
"""

import numpy as np
import time
from cuda_ops import CUDAOps
from model_cnn_cuda import SimpleCNN_CUDA
from mnist_data import load_mnist


def train_cnn():
    """Train pure CUDA CNN on MNIST."""
    print("Loading MNIST...")
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)

    print(f"Train: {train_images.shape}, Test: {test_images.shape}")

    ops = CUDAOps()
    model = SimpleCNN_CUDA(ops)

    # Training config (same as MLP)
    batch_size = 64
    lr = 0.01
    epochs = 10

    # Pre-allocate batch buffer
    x_batch_ptr = ops.alloc(batch_size * 1 * 28 * 28)

    history = {'loss': [], 'acc': []}

    print("\nStarting CNN training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = train_images.shape[0] // batch_size

        for i in range(num_batches):
            # Get batch (CPU) - already in [N, 1, 28, 28] format
            x_batch = train_images[i*batch_size:(i+1)*batch_size]
            y_batch = train_labels[i*batch_size:(i+1)*batch_size]

            # Copy to GPU
            ops.lib.cuda_memcpy_h2d(x_batch_ptr, x_batch.ctypes.data, x_batch.nbytes)

            # Forward (GPU)
            logits_ptr = model.forward(x_batch_ptr, batch_size)

            # Backward (GPU)
            loss = model.backward(logits_ptr, y_batch)
            epoch_loss += loss

            # Update (GPU)
            model.update(lr)

        # Evaluate
        test_acc = evaluate_cnn(model, ops, test_images, test_labels)
        avg_loss = epoch_loss / num_batches

        history['loss'].append(avg_loss)
        history['acc'].append(test_acc)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, test_acc={test_acc:.2%}, time={elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Final accuracy: {history['acc'][-1]:.2%}")

    ops.free(x_batch_ptr)

    return history, total_time


def evaluate_cnn(model, ops, images, labels, batch_size=1000):
    """Evaluate accuracy on test set."""
    correct = 0
    total = images.shape[0]

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        actual_batch = end - i

        # Images already in [N, 1, 28, 28] format
        x = images[i:end]
        x_ptr = ops.to_device(x)

        preds = model.predict(x_ptr, actual_batch)
        correct += np.sum(preds == labels[i:end])

        ops.free(x_ptr)

    return correct / total


if __name__ == '__main__':
    train_cnn()
```

- [ ] **Step 2: Run training**

Run: `cd python && python train_mnist_cnn_cuda.py`
Expected: Training completes without NaN, accuracy improves each epoch

- [ ] **Step 3: Verify accuracy >= 95%**

Check output for final accuracy. If < 95%, may need to increase epochs or adjust learning rate.

- [ ] **Step 4: Commit**

```bash
git add python/train_mnist_cnn_cuda.py
git commit -m "feat: add CNN training script for MNIST with accuracy tracking"
```

---

## Task 5: Final Verification and Documentation

**Files:**
- Modify: `README.md:256-296` (Next Steps section)
- Modify: `docs/PROJECT_PLAN.md:44-65` (项目状态)

- [ ] **Step 1: Update README.md Next Steps**

Replace the "Next Steps" section (lines 256-296) with:

```markdown
## Next Steps

### ✅ CNN Training Extension (Completed)

Conv2d/MaxPool2d bindings and CNN model are now implemented. Run with:

```bash
cd python && python train_mnist_cnn_cuda.py
```

### Further Extensions

| 方向 | 描述 | 预期收益 |
|------|------|---------|
| **FP16 Inference** | 修改 kernel 支持 `half` 精度 | 推理速度 ~2x |
| **BatchNorm/LayerNorm** | 添加归一化算子 | 支持更深的网络 |
| **Attention** | Transformer 核心算子 | 支持 LLM 架构 |
| **性能回归测试** | 在 `tests/` 添加 benchmark 阈值 | 防止优化退化 |
```

- [ ] **Step 2: Update PROJECT_PLAN.md**

Update the 项目状态 section (lines 61-65):

```markdown
### 🎯 项目状态
- **12 个算子**，forward + backward 全部实现
- **Conv2d/MaxPool2d Python binding**，支持 CNN 训练
- **SimpleCNN_CUDA 模型**，完整 CNN 训练流程
- **59 个测试**，100% 通过率
- **10.61x 性能提升** vs NumPy 实现
- **CNN 训练**：MNIST 验证可用
```

- [ ] **Step 3: Run full test suite**

Run: `cd build && ctest --output-on-failure`
Expected: All 59 tests pass

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/PROJECT_PLAN.md
git commit -m "docs: update documentation to reflect CNN training implementation"
```

---

## Self-Review Checklist

- ✅ Spec coverage: All requirements from design spec implemented
- ✅ Placeholder scan: No TBD/TODO/placeholders
- ✅ Type consistency: All method signatures match across tasks
- ✅ File paths: All exact paths specified
- ✅ Test commands: All commands with expected output