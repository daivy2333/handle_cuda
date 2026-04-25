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

    print("\n" + "="*50)
    print("All binding tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_binding()