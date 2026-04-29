"""
Pure CUDA CNN Model - All computation on GPU
"""

import numpy as np
from cuda_ops import CUDAOps


class SimpleCNN_CUDA:
    """
    Pure CUDA 2-Conv CNN for MNIST.
    All forward/backward computation happens on GPU.
    Only weight initialization and data loading use numpy.

    Architecture:
        Input [batch, 1, 28, 28]
        -> Conv1(1->16, kernel=3x3, stride=1, padding=1) [batch, 16, 28, 28]
        -> ReLU (in-place)
        -> MaxPool1(2x2, stride=2) [batch, 16, 14, 14]
        -> Conv2(16->32, kernel=3x3, stride=1, padding=1) [batch, 32, 14, 14]
        -> ReLU (in-place)
        -> MaxPool2(2x2, stride=2) [batch, 32, 7, 7]
        -> Flatten [batch, 1568]
        -> FC(1568->10) [batch, 10]
        -> Softmax + CrossEntropy
    """

    def __init__(self, ops: CUDAOps):
        self.ops = ops
        np.random.seed(42)

        # Layer dimensions
        self.conv1_in_C = 1
        self.conv1_out_C = 16
        self.conv1_kernel = 3
        self.conv1_stride = 1
        self.conv1_pad = 1

        self.conv2_in_C = 16
        self.conv2_out_C = 32
        self.conv2_kernel = 3
        self.conv2_stride = 1
        self.conv2_pad = 1

        self.pool_kernel = 2
        self.pool_stride = 2

        # After Conv1+Pool: 28 -> 28 -> 14
        # After Conv2+Pool: 14 -> 14 -> 7
        self.fc_in = 32 * 7 * 7  # 1568
        self.fc_out = 10

        # Initialize weights on CPU (Xavier initialization)
        # Conv1: [16, 1, 3, 3]
        conv1_w_np = (np.random.randn(
            self.conv1_out_C, self.conv1_in_C, self.conv1_kernel, self.conv1_kernel
        ) * np.sqrt(2.0 / (self.conv1_in_C * self.conv1_kernel * self.conv1_kernel))).astype(np.float32)
        conv1_b_np = np.zeros(self.conv1_out_C, dtype=np.float32)

        # Conv2: [32, 16, 3, 3]
        conv2_w_np = (np.random.randn(
            self.conv2_out_C, self.conv2_in_C, self.conv2_kernel, self.conv2_kernel
        ) * np.sqrt(2.0 / (self.conv2_in_C * self.conv2_kernel * self.conv2_kernel))).astype(np.float32)
        conv2_b_np = np.zeros(self.conv2_out_C, dtype=np.float32)

        # FC: [1568, 10]
        fc_w_np = (np.random.randn(self.fc_in, self.fc_out) * np.sqrt(2.0 / self.fc_in)).astype(np.float32)
        fc_b_np = np.zeros(self.fc_out, dtype=np.float32)

        # Copy weights to GPU
        self.conv1_w_ptr = self.ops.to_device(conv1_w_np)
        self.conv1_b_ptr = self.ops.to_device(conv1_b_np)
        self.conv2_w_ptr = self.ops.to_device(conv2_w_np)
        self.conv2_b_ptr = self.ops.to_device(conv2_b_np)
        self.fc_w_ptr = self.ops.to_device(fc_w_np)
        self.fc_b_ptr = self.ops.to_device(fc_b_np)

        # Allocate gradient buffers on GPU
        self.g_conv1_w_ptr = self.ops.alloc(self.conv1_out_C * self.conv1_in_C * self.conv1_kernel * self.conv1_kernel)
        self.g_conv1_b_ptr = self.ops.alloc(self.conv1_out_C)
        self.g_conv2_w_ptr = self.ops.alloc(self.conv2_out_C * self.conv2_in_C * self.conv2_kernel * self.conv2_kernel)
        self.g_conv2_b_ptr = self.ops.alloc(self.conv2_out_C)
        self.g_fc_w_ptr = self.ops.alloc(self.fc_in * self.fc_out)
        self.g_fc_b_ptr = self.ops.alloc(self.fc_out)

        # Cache for backward pass (all GPU pointers)
        self.cache = {}

    def __del__(self):
        """Destructor to free all model-owned GPU allocations."""
        if hasattr(self, 'ops') and self.ops:
            # Free weights
            if hasattr(self, 'conv1_w_ptr'):
                self.ops.free(self.conv1_w_ptr)
            if hasattr(self, 'conv1_b_ptr'):
                self.ops.free(self.conv1_b_ptr)
            if hasattr(self, 'conv2_w_ptr'):
                self.ops.free(self.conv2_w_ptr)
            if hasattr(self, 'conv2_b_ptr'):
                self.ops.free(self.conv2_b_ptr)
            if hasattr(self, 'fc_w_ptr'):
                self.ops.free(self.fc_w_ptr)
            if hasattr(self, 'fc_b_ptr'):
                self.ops.free(self.fc_b_ptr)
            # Free gradient buffers
            if hasattr(self, 'g_conv1_w_ptr'):
                self.ops.free(self.g_conv1_w_ptr)
            if hasattr(self, 'g_conv1_b_ptr'):
                self.ops.free(self.g_conv1_b_ptr)
            if hasattr(self, 'g_conv2_w_ptr'):
                self.ops.free(self.g_conv2_w_ptr)
            if hasattr(self, 'g_conv2_b_ptr'):
                self.ops.free(self.g_conv2_b_ptr)
            if hasattr(self, 'g_fc_w_ptr'):
                self.ops.free(self.g_fc_w_ptr)
            if hasattr(self, 'g_fc_b_ptr'):
                self.ops.free(self.g_fc_b_ptr)
            # Free cached activations if any
            if hasattr(self, 'cache'):
                for key in ['conv1_out_ptr', 'conv1_relu_ptr', 'pool1_out_ptr', 'pool1_indices_ptr',
                           'conv2_out_ptr', 'conv2_relu_ptr', 'pool2_out_ptr', 'pool2_indices_ptr',
                           'flat_ptr']:
                    if key in self.cache and self.cache[key]:
                        try:
                            # pool indices are int32, need special handling
                            if 'indices' in key:
                                self.ops.lib.cuda_free(self.cache[key])
                            else:
                                self.ops.free(self.cache[key])
                        except:
                            pass

    def forward(self, x_ptr, batch):
        """Pure CUDA forward pass.

        Args:
            x_ptr: GPU pointer to input [batch, 1, 28, 28]
            batch: batch size

        Returns:
            logits_ptr: GPU pointer to logits [batch, 10]
        """
        H, W = 28, 28

        # Conv1: [batch, 1, 28, 28] -> [batch, 16, 28, 28]
        conv1_out_ptr = self.ops.conv2d(
            x_ptr, self.conv1_w_ptr, self.conv1_b_ptr,
            batch, self.conv1_in_C, H, W,
            self.conv1_out_C, self.conv1_kernel, self.conv1_kernel,
            self.conv1_stride, self.conv1_stride, self.conv1_pad, self.conv1_pad
        )

        # ReLU (in-place)
        self.ops.relu(conv1_out_ptr, batch * self.conv1_out_C * H * W)

        # MaxPool1: [batch, 16, 28, 28] -> [batch, 16, 14, 14]
        pool1_H = (H + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool1_W = (W + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool1_out_ptr, pool1_indices_ptr = self.ops.maxpool2d(
            conv1_out_ptr, batch, self.conv1_out_C, H, W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride
        )

        # Conv2: [batch, 16, 14, 14] -> [batch, 32, 14, 14]
        conv2_out_ptr = self.ops.conv2d(
            pool1_out_ptr, self.conv2_w_ptr, self.conv2_b_ptr,
            batch, self.conv2_in_C, pool1_H, pool1_W,
            self.conv2_out_C, self.conv2_kernel, self.conv2_kernel,
            self.conv2_stride, self.conv2_stride, self.conv2_pad, self.conv2_pad
        )

        # ReLU (in-place)
        self.ops.relu(conv2_out_ptr, batch * self.conv2_out_C * pool1_H * pool1_W)

        # MaxPool2: [batch, 32, 14, 14] -> [batch, 32, 7, 7]
        pool2_H = (pool1_H + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool2_W = (pool1_W + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool2_out_ptr, pool2_indices_ptr = self.ops.maxpool2d(
            conv2_out_ptr, batch, self.conv2_out_C, pool1_H, pool1_W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride
        )

        # Flatten: [batch, 32, 7, 7] -> [batch, 1568]
        flat_ptr = self.ops.flatten(pool2_out_ptr, batch, self.conv2_out_C, pool2_H, pool2_W)

        # FC: [batch, 1568] @ [1568, 10] + bias -> [batch, 10]
        fc_out_ptr = self.ops.matmul(flat_ptr, self.fc_w_ptr, batch, self.fc_out, self.fc_in)
        logits_ptr = self.ops.bias_add(fc_out_ptr, self.fc_b_ptr, batch, self.fc_out)

        # Cache for backward (all GPU pointers)
        self.cache['x_ptr'] = x_ptr
        self.cache['conv1_relu_ptr'] = conv1_out_ptr  # post-relu output (acts as mask for relu_backward)
        self.cache['pool1_out_ptr'] = pool1_out_ptr
        self.cache['pool1_indices_ptr'] = pool1_indices_ptr
        self.cache['conv2_relu_ptr'] = conv2_out_ptr  # post-relu output
        self.cache['pool2_out_ptr'] = pool2_out_ptr
        self.cache['pool2_indices_ptr'] = pool2_indices_ptr
        self.cache['flat_ptr'] = flat_ptr
        self.cache['batch'] = batch

        # Free intermediate buffers (not needed for backward)
        self.ops.free(fc_out_ptr)

        return logits_ptr

    def backward(self, logits_ptr, targets):
        """Pure CUDA backward pass.

        Args:
            logits_ptr: GPU pointer to logits
            targets: numpy int32 array [batch] (stays on CPU)

        Returns:
            loss: scalar loss value
        """
        batch = self.cache['batch']

        # Cross entropy loss + gradient (returns GPU pointer)
        loss, grad_logits_ptr = self.ops.cross_entropy_loss(logits_ptr, targets, batch, self.fc_out)

        # Backprop FC layer: logits = flat @ fc_w + fc_b
        flat_ptr = self.cache['flat_ptr']

        # Compute gradients for FC
        grad_flat_ptr, grad_fc_w_ptr = self.ops.matmul_backward(
            grad_logits_ptr, flat_ptr, self.fc_w_ptr, batch, self.fc_out, self.fc_in,
            grad_A_ptr=None, grad_B_ptr=self.g_fc_w_ptr
        )

        # Bias gradient for FC
        _, grad_fc_b_ptr = self.ops.bias_add_backward(
            grad_logits_ptr, batch, self.fc_out,
            grad_input_ptr=None, grad_bias_ptr=self.g_fc_b_ptr
        )

        # Backprop flatten (identity in forward, so just reshape gradient)
        pool2_H, pool2_W = 7, 7
        grad_pool2_ptr = self.ops.flatten_backward(grad_flat_ptr, batch, self.conv2_out_C, pool2_H, pool2_W)

        # Backprop MaxPool2
        pool1_H, pool1_W = 14, 14
        pool2_indices_ptr = self.cache['pool2_indices_ptr']
        grad_conv2_relu_ptr = self.ops.maxpool2d_backward(
            grad_pool2_ptr, pool2_indices_ptr,
            batch, self.conv2_out_C, pool1_H, pool1_W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride,
            0, 0  # pad_h, pad_w
        )

        # Backprop ReLU (conv2 output)
        conv2_relu_ptr = self.cache['conv2_relu_ptr']
        grad_conv2_ptr = self.ops.alloc(batch * self.conv2_out_C * pool1_H * pool1_W)
        self.ops.relu_backward(grad_conv2_relu_ptr, conv2_relu_ptr, grad_conv2_ptr,
                               batch * self.conv2_out_C * pool1_H * pool1_W)

        # Backprop Conv2
        pool1_out_ptr = self.cache['pool1_out_ptr']
        grad_pool1_ptr, grad_conv2_w_ptr, grad_conv2_b_ptr = self.ops.conv2d_backward(
            grad_conv2_ptr, pool1_out_ptr, self.conv2_w_ptr,
            batch, self.conv2_in_C, pool1_H, pool1_W,
            self.conv2_out_C, self.conv2_kernel, self.conv2_kernel,
            self.conv2_stride, self.conv2_stride, self.conv2_pad, self.conv2_pad,
            grad_input_ptr=None, grad_weight_ptr=self.g_conv2_w_ptr, grad_bias_ptr=self.g_conv2_b_ptr
        )

        # Backprop MaxPool1
        H, W = 28, 28
        pool1_indices_ptr = self.cache['pool1_indices_ptr']
        grad_conv1_relu_ptr = self.ops.maxpool2d_backward(
            grad_pool1_ptr, pool1_indices_ptr,
            batch, self.conv1_out_C, H, W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride,
            0, 0  # pad_h, pad_w
        )

        # Backprop ReLU (conv1 output)
        conv1_relu_ptr = self.cache['conv1_relu_ptr']
        grad_conv1_ptr = self.ops.alloc(batch * self.conv1_out_C * H * W)
        self.ops.relu_backward(grad_conv1_relu_ptr, conv1_relu_ptr, grad_conv1_ptr,
                               batch * self.conv1_out_C * H * W)

        # Backprop Conv1
        x_ptr = self.cache['x_ptr']
        grad_x_ptr, grad_conv1_w_ptr, grad_conv1_b_ptr = self.ops.conv2d_backward(
            grad_conv1_ptr, x_ptr, self.conv1_w_ptr,
            batch, self.conv1_in_C, H, W,
            self.conv1_out_C, self.conv1_kernel, self.conv1_kernel,
            self.conv1_stride, self.conv1_stride, self.conv1_pad, self.conv1_pad,
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
        self.ops.free(grad_x_ptr)
        self.ops.free(logits_ptr)

        # Cleanup cached activations from forward pass
        for key in ['pool1_indices_ptr', 'pool2_indices_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.lib.cuda_free(self.cache[key])  # int32 indices
        for key in ['conv1_relu_ptr', 'pool1_out_ptr', 'conv2_relu_ptr', 'pool2_out_ptr', 'flat_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.free(self.cache[key])

        self.cache.clear()

        return loss

    def update(self, lr):
        """SGD update on GPU."""
        self.ops.sgd_update(self.conv1_w_ptr, self.g_conv1_w_ptr,
                           self.conv1_out_C * self.conv1_in_C * self.conv1_kernel * self.conv1_kernel, lr)
        self.ops.sgd_update(self.conv1_b_ptr, self.g_conv1_b_ptr, self.conv1_out_C, lr)
        self.ops.sgd_update(self.conv2_w_ptr, self.g_conv2_w_ptr,
                           self.conv2_out_C * self.conv2_in_C * self.conv2_kernel * self.conv2_kernel, lr)
        self.ops.sgd_update(self.conv2_b_ptr, self.g_conv2_b_ptr, self.conv2_out_C, lr)
        self.ops.sgd_update(self.fc_w_ptr, self.g_fc_w_ptr, self.fc_in * self.fc_out, lr)
        self.ops.sgd_update(self.fc_b_ptr, self.g_fc_b_ptr, self.fc_out, lr)

    def predict(self, x_ptr, batch):
        """Predict on GPU, return numpy array."""
        logits_ptr = self.forward(x_ptr, batch)
        logits = self.ops.to_host(logits_ptr, (batch, self.fc_out))
        # Free logits buffer
        self.ops.free(logits_ptr)
        # Clear cached activations (inference doesn't need backward)
        for key in ['pool1_indices_ptr', 'pool2_indices_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.lib.cuda_free(self.cache[key])  # int32 indices
        for key in ['conv1_relu_ptr', 'pool1_out_ptr', 'conv2_relu_ptr', 'pool2_out_ptr', 'flat_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.free(self.cache[key])
        self.cache.clear()
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
    assert logits.shape == (batch, 10), f"Expected shape (32, 10), got {logits.shape}"
    assert not np.any(np.isnan(logits)), "Logits contain NaN!"
    print("   Forward: PASSED")

    # Test backward
    print("\n2. Testing backward pass...")
    targets = np.random.randint(0, 10, batch).astype(np.int32)
    loss = model.backward(logits_ptr, targets)
    print(f"   Loss: {loss:.4f}")
    assert not np.isnan(loss), "Loss is NaN!"
    print("   Backward: PASSED")

    # Test update
    print("\n3. Testing update...")
    model.update(lr=0.01)
    print("   Update: PASSED")

    # Test predict
    print("\n4. Testing predict...")
    # Need to create new forward pass since previous logits were freed
    preds = model.predict(x_ptr, batch)
    assert preds.shape == (batch,), f"Expected shape (32,), got {preds.shape}"
    assert np.all(preds >= 0) and np.all(preds < 10), "Predictions out of range!"
    print(f"   Predictions: {preds[:5]}")
    print("   Predict: PASSED")

    # Test full training iteration
    print("\n5. Testing full training iteration...")
    x2 = np.random.randn(batch, 1, 28, 28).astype(np.float32)
    x2_ptr = ops.to_device(x2)
    targets2 = np.random.randint(0, 10, batch).astype(np.int32)

    logits2_ptr = model.forward(x2_ptr, batch)
    loss2 = model.backward(logits2_ptr, targets2)
    model.update(lr=0.01)
    print(f"   Loss after update: {loss2:.4f}")
    assert not np.isnan(loss2), "Loss is NaN after update!"
    print("   Full iteration: PASSED")

    ops.free(x_ptr)
    ops.free(x2_ptr)

    print("\n" + "=" * 50)
    print("All CUDA CNN model tests passed!")
    print("=" * 50)


if __name__ == '__main__':
    test_cnn_model()