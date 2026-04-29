# CNN Training Extension Design

## Overview

Extend the existing CUDA deep learning operators project to support CNN training on MNIST. The goal is to achieve 98%+ accuracy using a 2-Conv CNN architecture.

## Requirements

| Requirement | Description |
|-------------|-------------|
| CNN Architecture | Conv(1→16,3x3) → Pool → Conv(16→32,3x3) → Pool → FC → Softmax |
| Training Params | batch_size=64, epochs=10, lr=0.01 (same as MLP) |
| Verification | Accuracy validation only (expect 98%+) |
| Approach | Validate Conv2d/MaxPool2d bindings first, then integrate CNN model |

## Architecture

### CNN Data Flow

```
Input [N,1,28,28]
  → Conv1(1→16,k=3,s=1,p=1) [N,16,28,28]
  → ReLU
  → MaxPool1(2x2,s=2) [N,16,14,14]
  → Conv2(16→32,k=3,s=1,p=1) [N,32,14,14]
  → ReLU
  → MaxPool2(2x2,s=2) [N,32,7,7]
  → Flatten [N,1568]
  → FC(1568→10) [N,10]
  → Softmax + CrossEntropy
```

### Weight Dimensions

| Layer | Weight Shape | Bias Shape |
|-------|--------------|------------|
| Conv1 | [16, 1, 3, 3] | [16] |
| Conv2 | [32, 16, 3, 3] | [32] |
| FC | [1568, 10] | [10] |

## File Structure

```
python/
├── cuda_ops.py              # Add conv2d/maxpool2d bindings + tests
├── model_cnn_cuda.py        # NEW: SimpleCNN_CUDA class
└── train_mnist_cnn_cuda.py  # NEW: CNN training script

src/
└── cuda_ops_export.cu       # Add conv2d/maxpool2d C API exports
```

## Component Design

### 1. C API Export (cuda_ops_export.cu)

Add extern "C" functions:

```cpp
// Conv2d forward
void cuda_conv2d_f32(const float* input, const float* weight, const float* bias,
                     float* output, int N, int C, int H, int W,
                     int out_C, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int pad_h, int pad_w);

// Conv2d backward
void cuda_conv2d_backward_f32(const float* grad_out, const float* input,
                              const float* weight, float* grad_input,
                              float* grad_weight, float* grad_bias,
                              int N, int C, int H, int W,
                              int out_C, int kernel_h, int kernel_w,
                              int stride_h, int stride_w, int pad_h, int pad_w);

// MaxPool2d forward
void cuda_maxpool2d_f32(const float* input, float* output, int* indices,
                        int N, int C, int H, int W,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w, int pad_h, int pad_w);

// MaxPool2d backward
void cuda_maxpool2d_backward_f32(const float* grad_out, const int* indices,
                                  float* grad_input, int N, int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int stride_h, int stride_w, int pad_h, int pad_w);
```

### 2. Python Binding (cuda_ops.py)

Add methods to CUDAOps class:

```python
def conv2d(self, input_ptr, weight_ptr, bias_ptr,
           N, C, H, W, out_C, kernel_h, kernel_w,
           stride_h=1, stride_w=1, pad_h=0, pad_w=0,
           output_ptr=None):
    """Conv2d forward pass.

    Returns:
        output_ptr: GPU pointer to output [N, out_C, out_H, out_W]
    """

def conv2d_backward(self, grad_out_ptr, input_ptr, weight_ptr,
                    N, C, H, W, out_C, kernel_h, kernel_w,
                    stride_h, stride_w, pad_h, pad_w,
                    grad_input_ptr=None, grad_weight_ptr=None,
                    grad_bias_ptr=None):
    """Conv2d backward pass.

    Returns:
        grad_input_ptr, grad_weight_ptr, grad_bias_ptr
    """

def maxpool2d(self, input_ptr, N, C, H, W,
              kernel_h=2, kernel_w=2, stride_h=2, stride_w=2,
              pad_h=0, pad_w=0, output_ptr=None, indices_ptr=None):
    """MaxPool2d forward pass.

    Returns:
        output_ptr: GPU pointer to output [N, C, out_H, out_W]
        indices_ptr: GPU pointer to indices (for backward)
    """

def maxpool2d_backward(self, grad_out_ptr, indices_ptr, N, C, H, W,
                       kernel_h, kernel_w, stride_h, stride_w,
                       pad_h, pad_w, grad_input_ptr=None):
    """MaxPool2d backward pass.

    Returns:
        grad_input_ptr
    """
```

### 3. CNN Model (model_cnn_cuda.py)

```python
class SimpleCNN_CUDA:
    """Pure CUDA 2-Conv CNN for MNIST."""

    def __init__(self, ops: CUDAOps):
        self.ops = ops

        # Initialize weights (Xavier initialization)
        # Conv1: [16, 1, 3, 3], bias: [16]
        # Conv2: [32, 16, 3, 3], bias: [32]
        # FC: [1568, 10], bias: [10]

        # Allocate GPU memory and copy weights

        # Allocate gradient buffers

        # Cache for backward pass

    def forward(self, x_ptr, batch):
        """Forward pass: conv1 → relu → pool → conv2 → relu → pool → flatten → fc → softmax

        Args:
            x_ptr: GPU pointer [batch, 1, 28, 28]
            batch: batch size

        Returns:
            logits_ptr: GPU pointer [batch, 10]
        """

    def backward(self, logits_ptr, targets):
        """Backward pass.

        Args:
            logits_ptr: GPU pointer to logits
            targets: numpy int32 array [batch]

        Returns:
            loss: scalar loss value
        """

    def update(self, lr):
        """SGD update for all weights."""

    def predict(self, x_ptr, batch):
        """Predict on GPU, return numpy array."""
```

### 4. Training Script (train_mnist_cnn_cuda.py)

```python
def train_cnn():
    ops = CUDAOps()
    model = SimpleCNN_CUDA(ops)

    # Parameters (same as MLP)
    batch_size = 64
    epochs = 10
    lr = 0.01

    # Load MNIST data (reshape to [N, 1, 28, 28])

    # Training loop
    for epoch in range(epochs):
        # Forward + Backward + Update per batch

        # Print loss and test accuracy

    # Final accuracy should be 98%+
```

## Implementation Order

| Step | Task | Depends On |
|------|------|------------|
| 1 | Add Conv2d/MaxPool2d C API exports | None |
| 2 | Add Python bindings + binding tests | Step 1 |
| 3 | Build and verify bindings pass tests | Step 2 |
| 4 | Implement SimpleCNN_CUDA model | Step 3 |
| 5 | Implement training script | Step 4 |
| 6 | Run training, verify 98%+ accuracy | Step 5 |

## Testing Strategy

### Binding Tests (in cuda_ops.py)

```python
def test_conv2d_binding():
    # Compare CUDA conv2d output with PyTorch reference
    # Test forward and backward

def test_maxpool2d_binding():
    # Compare CUDA maxpool2d output with PyTorch reference
    # Test forward and backward
```

### Model Tests (in model_cnn_cuda.py)

```python
def test_cnn_model():
    # Test forward pass shape
    # Test backward pass (no NaN)
    # Test update
    # Test predict
```

## Success Criteria

| Criteria | Target |
|----------|--------|
| Binding tests | All pass, error < 1e-5 vs PyTorch |
| Model tests | All pass |
| Training accuracy | >= 98% on MNIST test set |
| Training time | Reasonable (similar to MLP ~8-10s) |

## Notes

- Conv2d and MaxPool2d CUDA kernels already exist and are verified by C++ tests
- Only need to export C API and add Python bindings
- Reuse existing ReLU, Softmax, CrossEntropy, Flatten, MatMul, BiasAdd, SGDUpdate
- Memory management: all GPU pointers freed in destructor