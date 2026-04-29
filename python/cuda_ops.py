"""
CUDA Operators Python Binding via ctypes
"""

import ctypes
import numpy as np
import os

class CUDAOps:
    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__),
                                     '..', 'build', 'lib', 'libcuda_ops_shared.so')
        self.lib = ctypes.CDLL(lib_path)
        self._setup_functions()

    def _setup_functions(self):
        # Memory
        self.lib.cuda_alloc.argtypes = [ctypes.c_size_t]
        self.lib.cuda_alloc.restype = ctypes.c_void_p

        self.lib.cuda_free.argtypes = [ctypes.c_void_p]
        self.lib.cuda_free.restype = None

        self.lib.cuda_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuda_memcpy_h2d.restype = None

        self.lib.cuda_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuda_memcpy_d2h.restype = None

        self.lib.cuda_sync.argtypes = []
        self.lib.cuda_sync.restype = None

        self.lib.cuda_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.lib.cuda_memset.restype = None

        # CrossEntropyLoss
        self.lib.cuda_cross_entropy_loss.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
        ]
        self.lib.cuda_cross_entropy_loss.restype = None

        # SGD Update
        self.lib.cuda_sgd_update.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p
        ]
        self.lib.cuda_sgd_update.restype = None

        # Flatten
        self.lib.cuda_flatten.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
        ]
        self.lib.cuda_flatten.restype = None

        self.lib.cuda_flatten_backward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
        ]
        self.lib.cuda_flatten_backward.restype = None

        # MatMul
        self.lib.cuda_matmul_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.cuda_matmul_f32.restype = None

        self.lib.cuda_matmul_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.cuda_matmul_backward_f32.restype = None

        # BiasAdd
        self.lib.cuda_bias_add_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.cuda_bias_add_f32.restype = None

        self.lib.cuda_bias_add_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.cuda_bias_add_backward_f32.restype = None

        # ReLU
        self.lib.cuda_relu_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuda_relu_f32.restype = None

        self.lib.cuda_relu_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        self.lib.cuda_relu_backward_f32.restype = None

        # Softmax
        self.lib.cuda_softmax_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.cuda_softmax_f32.restype = None

        self.lib.cuda_softmax_backward_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.cuda_softmax_backward_f32.restype = None

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

        # Conv2d backward with pre-allocated buffers (optimized)
        self.lib.cuda_conv2d_backward_with_buffers_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.cuda_conv2d_backward_with_buffers_f32.restype = None

        # Buffer size calculation
        self.lib.cuda_conv2d_backward_buffer_sizes_f32.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.cuda_conv2d_backward_buffer_sizes_f32.restype = ctypes.c_size_t

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

    def alloc(self, size):
        """Allocate GPU memory for size float32 elements."""
        return self.lib.cuda_alloc(size * 4)  # float32 = 4 bytes

    def free(self, ptr):
        """Free GPU memory."""
        self.lib.cuda_free(ptr)

    def to_device(self, arr):
        """Copy numpy array to GPU, returns GPU pointer."""
        ptr = self.alloc(arr.size)
        self.lib.cuda_memcpy_h2d(ptr, arr.ctypes.data, arr.nbytes)
        return ptr

    def to_host(self, ptr, shape):
        """Copy GPU memory to host numpy array."""
        result = np.empty(shape, dtype=np.float32)
        self.lib.cuda_memcpy_d2h(result.ctypes.data, ptr, result.nbytes)
        return result

    def sync(self):
        """Synchronize device."""
        self.lib.cuda_sync()

    def cross_entropy_loss(self, logits_ptr, targets, batch, classes):
        """Compute cross entropy loss and gradient.

        Args:
            logits_ptr: GPU pointer to logits (batch x classes)
            targets: numpy int32 array of target class indices (batch,)
            batch: batch size
            classes: number of classes

        Returns:
            loss: scalar loss value
            grad_ptr: GPU pointer to gradient
        """
        grad_ptr = self.alloc(batch * classes)
        # Allocate loss on device
        loss_ptr = self.alloc(1)
        # Set loss to 0 on device
        self.lib.cuda_memset(loss_ptr, 0, 4)  # 4 bytes for float32

        # Copy targets to device (kernel expects device pointer)
        targets_int = targets.astype(np.int32)
        targets_ptr = self.lib.cuda_alloc(targets_int.nbytes)
        self.lib.cuda_memcpy_h2d(targets_ptr, targets_int.ctypes.data, targets_int.nbytes)

        # stream = 0 (null pointer for default stream)
        self.lib.cuda_cross_entropy_loss(
            logits_ptr, targets_ptr,
            loss_ptr, grad_ptr, batch, classes, None
        )
        # Copy loss back to host
        loss = self.to_host(loss_ptr, (1,))
        self.free(loss_ptr)
        self.free(targets_ptr)
        return loss[0], grad_ptr

    def sgd_update(self, param_ptr, grad_ptr, size, lr):
        """Perform SGD update: param -= lr * grad.

        Args:
            param_ptr: GPU pointer to parameters
            grad_ptr: GPU pointer to gradients
            size: number of elements
            lr: learning rate
        """
        self.lib.cuda_sgd_update(param_ptr, grad_ptr, size, lr, None)

    def flatten(self, input_ptr, batch, C, H, W):
        """Flatten a 4D tensor to 2D.

        Args:
            input_ptr: GPU pointer to input (batch x C x H x W)
            batch, C, H, W: tensor dimensions

        Returns:
            output_ptr: GPU pointer to flattened output
        """
        output_ptr = self.alloc(batch * C * H * W)
        self.lib.cuda_flatten(input_ptr, output_ptr, batch, C, H, W, None)
        return output_ptr

    def flatten_backward(self, grad_flat_ptr, batch, C, H, W):
        """Backward pass for flatten.

        Args:
            grad_flat_ptr: GPU pointer to gradient from next layer
            batch, C, H, W: original tensor dimensions

        Returns:
            grad_input_ptr: GPU pointer to gradient in original shape
        """
        grad_input_ptr = self.alloc(batch * C * H * W)
        self.lib.cuda_flatten_backward(grad_flat_ptr, grad_input_ptr, batch, C, H, W, None)
        return grad_input_ptr

    def matmul(self, A_ptr, B_ptr, M, N, K, output_ptr=None):
        """Pure CUDA matmul: C = A @ B.

        Args:
            A_ptr: GPU pointer to A [M, K]
            B_ptr: GPU pointer to B [K, N]
            M, N, K: matrix dimensions
            output_ptr: optional pre-allocated output pointer

        Returns:
            C_ptr: GPU pointer to result [M, N]
        """
        if output_ptr is None:
            output_ptr = self.alloc(M * N)
        self.lib.cuda_matmul_f32(A_ptr, B_ptr, output_ptr, M, N, K)
        return output_ptr

    def matmul_backward(self, grad_C_ptr, A_ptr, B_ptr, M, N, K,
                        grad_A_ptr=None, grad_B_ptr=None):
        """Pure CUDA matmul backward.

        Returns:
            grad_A_ptr, grad_B_ptr
        """
        if grad_A_ptr is None:
            grad_A_ptr = self.alloc(M * K)
        if grad_B_ptr is None:
            grad_B_ptr = self.alloc(K * N)
        self.lib.cuda_matmul_backward_f32(
            grad_C_ptr, A_ptr, B_ptr, grad_A_ptr, grad_B_ptr, M, N, K)
        return grad_A_ptr, grad_B_ptr

    def bias_add(self, input_ptr, bias_ptr, rows, cols, output_ptr=None):
        """Pure CUDA bias add: output = input + bias.

        Args:
            input_ptr: GPU pointer to input [rows, cols]
            bias_ptr: GPU pointer to bias [cols]
            rows, cols: dimensions

        Returns:
            output_ptr
        """
        if output_ptr is None:
            output_ptr = self.alloc(rows * cols)
        self.lib.cuda_bias_add_f32(input_ptr, bias_ptr, output_ptr, rows, cols)
        return output_ptr

    def bias_add_backward(self, grad_out_ptr, rows, cols,
                          grad_input_ptr=None, grad_bias_ptr=None):
        """Pure CUDA bias add backward."""
        if grad_input_ptr is None:
            grad_input_ptr = self.alloc(rows * cols)
        if grad_bias_ptr is None:
            grad_bias_ptr = self.alloc(cols)
        self.lib.cuda_bias_add_backward_f32(
            grad_out_ptr, grad_input_ptr, grad_bias_ptr, rows, cols)
        return grad_input_ptr, grad_bias_ptr

    def relu(self, data_ptr, size):
        """Inplace ReLU activation."""
        self.lib.cuda_relu_f32(data_ptr, size)

    def relu_backward(self, grad_out_ptr, forward_input_ptr, grad_in_ptr, size):
        """ReLU backward."""
        self.lib.cuda_relu_backward_f32(
            grad_out_ptr, forward_input_ptr, grad_in_ptr, size)

    def softmax(self, input_ptr, batch, classes, output_ptr=None):
        """Pure CUDA softmax."""
        if output_ptr is None:
            output_ptr = self.alloc(batch * classes)
        self.lib.cuda_softmax_f32(input_ptr, output_ptr, batch, classes)
        return output_ptr

    def softmax_backward(self, grad_out_ptr, forward_output_ptr, batch, classes,
                         grad_in_ptr=None):
        """Softmax backward."""
        if grad_in_ptr is None:
            grad_in_ptr = self.alloc(batch * classes)
        self.lib.cuda_softmax_backward_f32(
            grad_out_ptr, forward_output_ptr, grad_in_ptr, batch, classes)
        return grad_in_ptr

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

    def conv2d_backward_with_buffers(self, grad_out_ptr, input_ptr, weight_ptr,
                                      N, C, H, W, out_C, kernel_h, kernel_w,
                                      stride_h, stride_w, pad_h, pad_w,
                                      reshaped_grad_ptr, col_buffer_ptr, col_grad_ptr,
                                      grad_input_ptr=None, grad_weight_ptr=None, grad_bias_ptr=None):
        """Optimized Conv2d backward with pre-allocated buffers (no malloc/free overhead).

        Args:
            grad_out_ptr: GPU pointer to gradient of output [N, out_C, out_H, out_W]
            input_ptr: GPU pointer to forward input [N, C, H, W]
            weight_ptr: GPU pointer to weight [out_C, C, kernel_h, kernel_w]
            reshaped_grad_ptr: Pre-allocated buffer [out_C, N*out_H*out_W]
            col_buffer_ptr: Pre-allocated buffer [C*K*K, N*out_H*out_W]
            col_grad_ptr: Pre-allocated buffer [C*K*K, N*out_H*out_W]
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

        self.lib.cuda_conv2d_backward_with_buffers_f32(
            grad_out_ptr, input_ptr, weight_ptr,
            grad_input_ptr, grad_weight_ptr, grad_bias_ptr,
            reshaped_grad_ptr, col_buffer_ptr, col_grad_ptr,
            N, C, H, W, out_C, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w
        )
        return grad_input_ptr, grad_weight_ptr, grad_bias_ptr

    def conv2d_backward_buffer_sizes(self, N, C, H, W, out_C, kernel_h, kernel_w,
                                      stride_h, stride_w, pad_h, pad_w):
        """Calculate buffer sizes needed for conv2d_backward_with_buffers.

        Args:
            Same dimensions as conv2d forward

        Returns:
            (total_size, reshaped_size, col_size) in bytes
        """
        reshaped_size = ctypes.c_size_t()
        col_size = ctypes.c_size_t()

        total_size = self.lib.cuda_conv2d_backward_buffer_sizes_f32(
            N, C, H, W, out_C, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            ctypes.byref(reshaped_size), ctypes.byref(col_size)
        )

        return total_size, reshaped_size.value, col_size.value

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


def test_binding():
    print("Testing CUDA Operators Python binding...")

    ops = CUDAOps()

    # Test memory operations
    print("\n1. Testing memory operations...")
    data = np.random.randn(32, 10).astype(np.float32)
    ptr = ops.to_device(data)
    result = ops.to_host(ptr, (32, 10))
    assert np.allclose(data, result), "Memory copy test failed!"
    ops.free(ptr)
    print("   Memory operations: PASSED")

    # Test cross entropy loss
    print("\n2. Testing cross entropy loss...")
    logits = np.random.randn(32, 10).astype(np.float32)
    targets = np.random.randint(0, 10, 32).astype(np.int32)

    logits_ptr = ops.to_device(logits)
    loss, grad_ptr = ops.cross_entropy_loss(logits_ptr, targets, 32, 10)

    print(f"   Loss: {loss:.4f}")
    grad = ops.to_host(grad_ptr, (32, 10))
    print(f"   Gradient shape: {grad.shape}")

    # Verify gradient is reasonable
    assert grad.shape == (32, 10), "Gradient shape mismatch!"
    assert not np.isnan(loss), "Loss is NaN!"
    assert not np.any(np.isnan(grad)), "Gradient contains NaN!"

    ops.free(logits_ptr)
    ops.free(grad_ptr)
    print("   Cross entropy loss: PASSED")

    # Test SGD update
    print("\n3. Testing SGD update...")
    param = np.ones(10, dtype=np.float32)
    grad = np.ones(10, dtype=np.float32) * 0.1

    param_ptr = ops.to_device(param)
    grad_ptr = ops.to_device(grad)

    ops.sgd_update(param_ptr, grad_ptr, 10, 0.01)
    ops.sync()

    updated_param = ops.to_host(param_ptr, (10,))
    expected = param - 0.01 * grad
    assert np.allclose(updated_param, expected), "SGD update test failed!"

    ops.free(param_ptr)
    ops.free(grad_ptr)
    print("   SGD update: PASSED")

    # Test flatten
    print("\n4. Testing flatten...")
    input_data = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_ptr = ops.to_device(input_data)

    flat_ptr = ops.flatten(input_ptr, 2, 3, 4, 4)
    flat_result = ops.to_host(flat_ptr, (2, 48))

    expected_flat = input_data.reshape(2, -1)
    assert np.allclose(flat_result, expected_flat), "Flatten test failed!"
    print(f"   Flattened shape: {input_data.shape} -> {flat_result.shape}")

    ops.free(input_ptr)
    ops.free(flat_ptr)
    print("   Flatten: PASSED")

    # Test matmul
    print("\n5. Testing matmul...")
    M, K, N = 4, 3, 5
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    A_ptr = ops.to_device(A)
    B_ptr = ops.to_device(B)

    C_ptr = ops.matmul(A_ptr, B_ptr, M, N, K)
    C_result = ops.to_host(C_ptr, (M, N))

    expected = A @ B
    assert np.allclose(C_result, expected, atol=1e-5), "Matmul test failed!"
    print(f"   Matmul shape: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")

    # Test matmul backward
    grad_C = np.random.randn(M, N).astype(np.float32)
    grad_C_ptr = ops.to_device(grad_C)
    grad_A_ptr, grad_B_ptr = ops.matmul_backward(grad_C_ptr, A_ptr, B_ptr, M, N, K)

    grad_A_result = ops.to_host(grad_A_ptr, (M, K))
    grad_B_result = ops.to_host(grad_B_ptr, (K, N))

    # Verify gradients: dL/dA = dL/dC @ B.T, dL/dB = A.T @ dL/dC
    expected_grad_A = grad_C @ B.T
    expected_grad_B = A.T @ grad_C
    assert np.allclose(grad_A_result, expected_grad_A, atol=1e-5), "Matmul backward grad_A failed!"
    assert np.allclose(grad_B_result, expected_grad_B, atol=1e-5), "Matmul backward grad_B failed!"

    ops.free(A_ptr)
    ops.free(B_ptr)
    ops.free(C_ptr)
    ops.free(grad_C_ptr)
    ops.free(grad_A_ptr)
    ops.free(grad_B_ptr)
    print("   Matmul and backward: PASSED")

    # Test bias_add
    print("\n6. Testing bias_add...")
    rows, cols = 8, 4
    input_data = np.random.randn(rows, cols).astype(np.float32)
    bias = np.random.randn(cols).astype(np.float32)

    input_ptr = ops.to_device(input_data)
    bias_ptr = ops.to_device(bias)

    output_ptr = ops.bias_add(input_ptr, bias_ptr, rows, cols)
    output_result = ops.to_host(output_ptr, (rows, cols))

    expected = input_data + bias
    assert np.allclose(output_result, expected, atol=1e-5), "Bias add test failed!"

    # Test bias_add backward
    grad_out = np.random.randn(rows, cols).astype(np.float32)
    grad_out_ptr = ops.to_device(grad_out)
    grad_input_ptr, grad_bias_ptr = ops.bias_add_backward(grad_out_ptr, rows, cols)

    grad_input_result = ops.to_host(grad_input_ptr, (rows, cols))
    grad_bias_result = ops.to_host(grad_bias_ptr, (cols,))

    expected_grad_input = grad_out
    expected_grad_bias = grad_out.sum(axis=0)
    assert np.allclose(grad_input_result, expected_grad_input, atol=1e-5), "Bias add backward grad_input failed!"
    assert np.allclose(grad_bias_result, expected_grad_bias, atol=1e-5), "Bias add backward grad_bias failed!"

    ops.free(input_ptr)
    ops.free(bias_ptr)
    ops.free(output_ptr)
    ops.free(grad_out_ptr)
    ops.free(grad_input_ptr)
    ops.free(grad_bias_ptr)
    print("   Bias add and backward: PASSED")

    # Test relu
    print("\n7. Testing relu...")
    size = 100
    relu_input = np.random.randn(size).astype(np.float32)

    # Preserve original input for backward pass
    forward_input_ptr = ops.to_device(relu_input.copy())  # Original input for backward
    relu_ptr = ops.to_device(relu_input.copy())  # Buffer for ReLU output (in-place)

    ops.relu(relu_ptr, size)
    relu_result = ops.to_host(relu_ptr, (size,))

    expected_relu = np.maximum(relu_input, 0)
    assert np.allclose(relu_result, expected_relu, atol=1e-5), "ReLU test failed!"

    # Test relu backward (use forward_input_ptr, not relu_ptr which contains output)
    grad_relu = np.random.randn(size).astype(np.float32)
    grad_relu_ptr = ops.to_device(grad_relu)
    grad_in_ptr = ops.alloc(size)

    ops.relu_backward(grad_relu_ptr, forward_input_ptr, grad_in_ptr, size)
    grad_in_result = ops.to_host(grad_in_ptr, (size,))

    expected_grad_in = grad_relu * (relu_input > 0).astype(np.float32)
    assert np.allclose(grad_in_result, expected_grad_in, atol=1e-5), "ReLU backward test failed!"

    ops.free(forward_input_ptr)
    ops.free(relu_ptr)
    ops.free(grad_relu_ptr)
    ops.free(grad_in_ptr)
    print("   ReLU and backward: PASSED")

    # Test softmax
    print("\n8. Testing softmax...")
    batch, classes = 4, 10
    softmax_input = np.random.randn(batch, classes).astype(np.float32)
    softmax_ptr = ops.to_device(softmax_input)

    softmax_out_ptr = ops.softmax(softmax_ptr, batch, classes)
    softmax_result = ops.to_host(softmax_out_ptr, (batch, classes))

    # Verify softmax: each row sums to 1, all values in (0, 1)
    row_sums = softmax_result.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Softmax rows don't sum to 1!"
    assert np.all(softmax_result >= 0) and np.all(softmax_result <= 1), "Softmax values out of range!"

    # Test softmax backward
    grad_softmax = np.random.randn(batch, classes).astype(np.float32)
    grad_softmax_ptr = ops.to_device(grad_softmax)
    grad_soft_in_ptr = ops.softmax_backward(grad_softmax_ptr, softmax_out_ptr, batch, classes)

    grad_soft_in_result = ops.to_host(grad_soft_in_ptr, (batch, classes))
    assert grad_soft_in_result.shape == (batch, classes), "Softmax backward shape mismatch!"
    assert not np.any(np.isnan(grad_soft_in_result)), "Softmax backward contains NaN!"

    # Numerical verification of softmax backward formula:
    # grad_in = softmax_output * (grad_out - sum(grad_out * softmax_output, axis=1, keepdims=True))
    softmax_output = softmax_result  # Already computed softmax output
    expected_grad_soft_in = softmax_output * (
        grad_softmax - np.sum(grad_softmax * softmax_output, axis=1, keepdims=True)
    )
    assert np.allclose(grad_soft_in_result, expected_grad_soft_in, atol=1e-5), \
        "Softmax backward numerical verification failed!"

    ops.free(softmax_ptr)
    ops.free(softmax_out_ptr)
    ops.free(grad_softmax_ptr)
    ops.free(grad_soft_in_ptr)
    print("   Softmax and backward: PASSED")

    # Test conv2d
    print("\n9. Testing conv2d...")
    import torch
    import torch.nn.functional as F

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

    # Test maxpool2d
    print("\n10. Testing maxpool2d...")
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
    # Note: indices are int32, so we use cuda_free directly (not self.free which assumes float32)
    ops.lib.cuda_free(indices_ptr)
    print("   MaxPool2d forward: PASSED")

    # Test conv2d backward
    print("\n11. Testing conv2d backward...")
    N, C, H, W = 2, 1, 28, 28
    out_C, kernel = 16, 3
    stride, pad = 1, 1

    input_np = np.random.randn(N, C, H, W).astype(np.float32)
    weight_np = np.random.randn(out_C, C, kernel, kernel).astype(np.float32)
    bias_np = np.random.randn(out_C).astype(np.float32)

    # PyTorch reference with autograd
    input_torch = torch.from_numpy(input_np).requires_grad_(True)
    weight_torch = torch.from_numpy(weight_np).requires_grad_(True)
    bias_torch = torch.from_numpy(bias_np).requires_grad_(True)
    output_torch = F.conv2d(input_torch, weight_torch, bias_torch, stride=stride, padding=pad)
    grad_out_torch = torch.randn_like(output_torch)
    output_torch.backward(grad_out_torch)

    # CUDA backward
    input_ptr = ops.to_device(input_np)
    weight_ptr = ops.to_device(weight_np)
    bias_ptr = ops.to_device(bias_np)
    output_ptr = ops.conv2d(input_ptr, weight_ptr, bias_ptr, N, C, H, W, out_C, kernel, kernel, stride, stride, pad, pad)
    grad_out_ptr = ops.to_device(grad_out_torch.detach().numpy())

    grad_in_ptr, grad_w_ptr, grad_b_ptr = ops.conv2d_backward(grad_out_ptr, input_ptr, weight_ptr, N, C, H, W, out_C, kernel, kernel, stride, stride, pad, pad)

    # Compare
    grad_in_result = ops.to_host(grad_in_ptr, input_np.shape)
    grad_w_result = ops.to_host(grad_w_ptr, weight_np.shape)
    grad_b_result = ops.to_host(grad_b_ptr, (out_C,))

    max_diff_in = np.abs(grad_in_result - input_torch.grad.numpy()).max()
    max_diff_w = np.abs(grad_w_result - weight_torch.grad.numpy()).max()
    max_diff_b = np.abs(grad_b_result - bias_torch.grad.numpy()).max()

    print(f"   Max diff grad_input vs PyTorch: {max_diff_in:.6f}")
    print(f"   Max diff grad_weight vs PyTorch: {max_diff_w:.6f}")
    print(f"   Max diff grad_bias vs PyTorch: {max_diff_b:.6f}")

    assert max_diff_in < 1e-3, f"Conv2d backward grad_input mismatch: {max_diff_in}"
    assert max_diff_w < 1e-3, f"Conv2d backward grad_weight mismatch: {max_diff_w}"
    assert max_diff_b < 1e-3, f"Conv2d backward grad_bias mismatch: {max_diff_b}"

    ops.free(input_ptr)
    ops.free(weight_ptr)
    ops.free(bias_ptr)
    ops.free(output_ptr)
    ops.free(grad_out_ptr)
    ops.free(grad_in_ptr)
    ops.free(grad_w_ptr)
    ops.free(grad_b_ptr)
    print("   Conv2d backward: PASSED")

    # Test conv2d with None bias
    print("\n12. Testing conv2d with None bias...")
    N, C, H, W = 2, 1, 28, 28
    out_C, kernel = 16, 3
    stride, pad = 1, 1

    input_np = np.random.randn(N, C, H, W).astype(np.float32)
    weight_np = np.random.randn(out_C, C, kernel, kernel).astype(np.float32)

    # PyTorch reference without bias
    input_torch = torch.from_numpy(input_np)
    weight_torch = torch.from_numpy(weight_np)
    output_torch = F.conv2d(input_torch, weight_torch, bias=None, stride=stride, padding=pad)
    output_np_ref = output_torch.numpy()

    # CUDA implementation with None bias
    input_ptr = ops.to_device(input_np)
    weight_ptr = ops.to_device(weight_np)

    output_ptr = ops.conv2d(input_ptr, weight_ptr, None, N, C, H, W, out_C, kernel, kernel, stride, stride, pad, pad)
    output_np = ops.to_host(output_ptr, (N, out_C, 28, 28))

    max_diff = np.abs(output_np - output_np_ref).max()
    print(f"   Max diff vs PyTorch (no bias): {max_diff:.6f}")
    assert max_diff < 1e-5, f"Conv2d (no bias) output mismatch: {max_diff}"

    ops.free(input_ptr)
    ops.free(weight_ptr)
    ops.free(output_ptr)
    print("   Conv2d with None bias: PASSED")

    # Test maxpool2d backward
    print("\n13. Testing maxpool2d backward...")
    N, C, H, W = 2, 16, 28, 28
    kernel_h, kernel_w = 2, 2
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 0, 0

    input_np = np.random.randn(N, C, H, W).astype(np.float32)

    # PyTorch reference with autograd
    input_torch = torch.from_numpy(input_np).requires_grad_(True)
    output_torch = F.max_pool2d(input_torch, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    grad_out_torch = torch.randn_like(output_torch)
    output_torch.backward(grad_out_torch)

    # CUDA forward + backward
    input_ptr = ops.to_device(input_np)
    output_ptr, indices_ptr = ops.maxpool2d(input_ptr, N, C, H, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

    out_H = (H + 2 * pad_h - kernel_h) // stride_h + 1
    out_W = (W + 2 * pad_w - kernel_w) // stride_w + 1
    grad_out_ptr = ops.to_device(grad_out_torch.detach().numpy())
    grad_in_ptr = ops.maxpool2d_backward(grad_out_ptr, indices_ptr, N, C, H, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

    grad_in_result = ops.to_host(grad_in_ptr, input_np.shape)

    max_diff = np.abs(grad_in_result - input_torch.grad.numpy()).max()
    print(f"   Max diff grad_input vs PyTorch: {max_diff:.6f}")
    assert max_diff < 1e-5, f"MaxPool2d backward mismatch: {max_diff}"

    ops.free(input_ptr)
    ops.free(output_ptr)
    ops.lib.cuda_free(indices_ptr)
    ops.free(grad_out_ptr)
    ops.free(grad_in_ptr)
    print("   MaxPool2d backward: PASSED")

    print("\n" + "="*50)
    print("All binding tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_binding()