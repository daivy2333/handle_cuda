"""
Pure CUDA CNN Model with cuBLAS Backend - All computation on GPU
"""

import numpy as np
from cuda_ops import CUDAOps


class SimpleCNN_CUBLAS:
    """
    Pure CUDA 2-Conv CNN for MNIST using cuBLAS backend.
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

    def __init__(self, ops: CUDAOps, batch_size=64):
        self.ops = ops
        self.batch_size = batch_size
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
        conv1_w_np = (np.random.randn(
            self.conv1_out_C, self.conv1_in_C, self.conv1_kernel, self.conv1_kernel
        ) * np.sqrt(2.0 / (self.conv1_in_C * self.conv1_kernel * self.conv1_kernel))).astype(np.float32)
        conv1_b_np = np.zeros(self.conv1_out_C, dtype=np.float32)

        conv2_w_np = (np.random.randn(
            self.conv2_out_C, self.conv2_in_C, self.conv2_kernel, self.conv2_kernel
        ) * np.sqrt(2.0 / (self.conv2_in_C * self.conv2_kernel * self.conv2_kernel))).astype(np.float32)
        conv2_b_np = np.zeros(self.conv2_out_C, dtype=np.float32)

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

        # Pre-allocate buffers for cuBLAS conv2d backward
        max_batch = batch_size
        H, W = 28, 28

        # Conv1 buffer sizes: N=64, C=1, out_C=16, kernel=3
        out_H1 = (H + 2 * self.conv1_pad - self.conv1_kernel) // self.conv1_stride + 1
        out_W1 = (W + 2 * self.conv1_pad - self.conv1_kernel) // self.conv1_stride + 1
        conv1_col_rows = self.conv1_in_C * self.conv1_kernel * self.conv1_kernel
        conv1_col_cols = max_batch * out_H1 * out_W1

        self.conv1_col_buffer_ptr = self.ops.alloc(conv1_col_rows * conv1_col_cols)
        self.conv1_grad_col_buffer_ptr = self.ops.alloc(conv1_col_rows * conv1_col_cols)
        self.conv1_grad_gemm_buffer_ptr = self.ops.alloc(self.conv1_out_C * conv1_col_cols)

        # Conv2 buffer sizes: N=64, C=16, out_C=32, kernel=3, H=14, W=14
        pool1_H, pool1_W = 14, 14
        out_H2 = (pool1_H + 2 * self.conv2_pad - self.conv2_kernel) // self.conv2_stride + 1
        out_W2 = (pool1_W + 2 * self.conv2_pad - self.conv2_kernel) // self.conv2_stride + 1
        conv2_col_rows = self.conv2_in_C * self.conv2_kernel * self.conv2_kernel
        conv2_col_cols = max_batch * out_H2 * out_W2

        self.conv2_col_buffer_ptr = self.ops.alloc(conv2_col_rows * conv2_col_cols)
        self.conv2_grad_col_buffer_ptr = self.ops.alloc(conv2_col_rows * conv2_col_cols)
        self.conv2_grad_gemm_buffer_ptr = self.ops.alloc(self.conv2_out_C * conv2_col_cols)

        # Pre-allocate forward buffers (col_buffer and gemm_buffer reused)
        self.conv1_gemm_buffer_ptr = self.ops.alloc(self.conv1_out_C * conv1_col_cols)
        self.conv2_gemm_buffer_ptr = self.ops.alloc(self.conv2_out_C * conv2_col_cols)

        # Cache for backward pass
        self.cache = {}

    def __del__(self):
        """Destructor to free all model-owned GPU allocations."""
        if hasattr(self, 'ops') and self.ops:
            # Free weights
            for attr in ['conv1_w_ptr', 'conv1_b_ptr', 'conv2_w_ptr', 'conv2_b_ptr',
                        'fc_w_ptr', 'fc_b_ptr']:
                if hasattr(self, attr):
                    try:
                        self.ops.free(getattr(self, attr))
                    except:
                        pass
            # Free gradient buffers
            for attr in ['g_conv1_w_ptr', 'g_conv1_b_ptr', 'g_conv2_w_ptr', 'g_conv2_b_ptr',
                        'g_fc_w_ptr', 'g_fc_b_ptr']:
                if hasattr(self, attr):
                    try:
                        self.ops.free(getattr(self, attr))
                    except:
                        pass
            # Free pre-allocated buffers
            for attr in ['conv1_col_buffer_ptr', 'conv1_grad_col_buffer_ptr',
                        'conv1_grad_gemm_buffer_ptr', 'conv2_col_buffer_ptr',
                        'conv2_grad_col_buffer_ptr', 'conv2_grad_gemm_buffer_ptr',
                        'conv1_gemm_buffer_ptr', 'conv2_gemm_buffer_ptr']:
                if hasattr(self, attr):
                    try:
                        self.ops.free(getattr(self, attr))
                    except:
                        pass
            # Free cached activations
            if hasattr(self, 'cache'):
                for key in ['pool1_indices_ptr', 'pool2_indices_ptr']:
                    if key in self.cache and self.cache[key]:
                        try:
                            self.ops.lib.cuda_free(self.cache[key])
                        except:
                            pass
                for key in ['conv1_pre_relu_ptr', 'conv1_relu_ptr', 'pool1_out_ptr',
                           'conv2_pre_relu_ptr', 'conv2_relu_ptr', 'pool2_out_ptr', 'flat_ptr']:
                    if key in self.cache and self.cache[key]:
                        try:
                            self.ops.free(self.cache[key])
                        except:
                            pass

    def forward(self, x_ptr, batch):
        """Pure CUDA forward pass using cuBLAS backend.

        Args:
            x_ptr: GPU pointer to input [batch, 1, 28, 28]
            batch: batch size

        Returns:
            logits_ptr: GPU pointer to logits [batch, 10]
        """
        H, W = 28, 28

        # Conv1 using cuBLAS: [batch, 1, 28, 28] -> [batch, 16, 28, 28]
        # conv1_out_ptr contains pre-ReLU values
        conv1_pre_relu_ptr = self.ops.conv2d_cublas(
            x_ptr, self.conv1_w_ptr, self.conv1_b_ptr,
            batch, self.conv1_in_C, H, W,
            self.conv1_out_C, self.conv1_kernel, self.conv1_kernel,
            self.conv1_stride, self.conv1_stride, self.conv1_pad, self.conv1_pad,
            col_buffer=self.conv1_col_buffer_ptr,
            gemm_buffer=self.conv1_gemm_buffer_ptr
        )

        # Out-of-place ReLU: input preserved for backward mask, output for next layer
        conv1_relu_ptr = self.ops.alloc(batch * self.conv1_out_C * H * W)
        self.ops.relu_out_of_place(conv1_pre_relu_ptr, conv1_relu_ptr,
                                    batch * self.conv1_out_C * H * W)

        # MaxPool1: [batch, 16, 28, 28] -> [batch, 16, 14, 14]
        pool1_H = (H + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool1_W = (W + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool1_out_ptr, pool1_indices_ptr = self.ops.maxpool2d(
            conv1_relu_ptr, batch, self.conv1_out_C, H, W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride
        )

        # Conv2 using cuBLAS: [batch, 16, 14, 14] -> [batch, 32, 14, 14]
        # conv2_out_ptr contains pre-ReLU values
        conv2_pre_relu_ptr = self.ops.conv2d_cublas(
            pool1_out_ptr, self.conv2_w_ptr, self.conv2_b_ptr,
            batch, self.conv2_in_C, pool1_H, pool1_W,
            self.conv2_out_C, self.conv2_kernel, self.conv2_kernel,
            self.conv2_stride, self.conv2_stride, self.conv2_pad, self.conv2_pad,
            col_buffer=self.conv2_col_buffer_ptr,
            gemm_buffer=self.conv2_gemm_buffer_ptr
        )

        # Out-of-place ReLU: input preserved for backward mask, output for next layer
        conv2_relu_ptr = self.ops.alloc(batch * self.conv2_out_C * pool1_H * pool1_W)
        self.ops.relu_out_of_place(conv2_pre_relu_ptr, conv2_relu_ptr,
                                    batch * self.conv2_out_C * pool1_H * pool1_W)

        # MaxPool2: [batch, 32, 14, 14] -> [batch, 32, 7, 7]
        pool2_H = (pool1_H + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool2_W = (pool1_W + 2 * 0 - self.pool_kernel) // self.pool_stride + 1
        pool2_out_ptr, pool2_indices_ptr = self.ops.maxpool2d(
            conv2_relu_ptr, batch, self.conv2_out_C, pool1_H, pool1_W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride
        )

        # Flatten: [batch, 32, 7, 7] -> [batch, 1568]
        flat_ptr = self.ops.flatten(pool2_out_ptr, batch, self.conv2_out_C, pool2_H, pool2_W)

        # FC: [batch, 1568] @ [1568, 10] + bias -> [batch, 10]
        fc_out_ptr = self.ops.matmul(flat_ptr, self.fc_w_ptr, batch, self.fc_out, self.fc_in)
        logits_ptr = self.ops.bias_add(fc_out_ptr, self.fc_b_ptr, batch, self.fc_out)

        # Cache for backward (pre-ReLU values as mask)
        self.cache['x_ptr'] = x_ptr
        self.cache['conv1_pre_relu_ptr'] = conv1_pre_relu_ptr
        self.cache['conv1_relu_ptr'] = conv1_relu_ptr
        self.cache['pool1_out_ptr'] = pool1_out_ptr
        self.cache['pool1_indices_ptr'] = pool1_indices_ptr
        self.cache['conv2_pre_relu_ptr'] = conv2_pre_relu_ptr
        self.cache['conv2_relu_ptr'] = conv2_relu_ptr
        self.cache['pool2_out_ptr'] = pool2_out_ptr
        self.cache['pool2_indices_ptr'] = pool2_indices_ptr
        self.cache['flat_ptr'] = flat_ptr
        self.cache['batch'] = batch

        # Free intermediate buffers
        self.ops.free(fc_out_ptr)

        return logits_ptr

    def backward(self, logits_ptr, targets):
        """Pure CUDA backward pass using cuBLAS backend.

        Args:
            logits_ptr: GPU pointer to logits
            targets: numpy int32 array [batch]

        Returns:
            loss: scalar loss value
        """
        batch = self.cache['batch']
        pool1_H, pool1_W = 14, 14
        pool2_H, pool2_W = 7, 7
        H, W = 28, 28

        # Cross entropy loss + gradient
        loss, grad_logits_ptr = self.ops.cross_entropy_loss(logits_ptr, targets, batch, self.fc_out)

        # Backprop FC layer
        flat_ptr = self.cache['flat_ptr']
        grad_flat_ptr, grad_fc_w_ptr = self.ops.matmul_backward(
            grad_logits_ptr, flat_ptr, self.fc_w_ptr, batch, self.fc_out, self.fc_in,
            grad_A_ptr=None, grad_B_ptr=self.g_fc_w_ptr
        )
        _, grad_fc_b_ptr = self.ops.bias_add_backward(
            grad_logits_ptr, batch, self.fc_out,
            grad_input_ptr=None, grad_bias_ptr=self.g_fc_b_ptr
        )

        # Backprop flatten
        grad_pool2_ptr = self.ops.flatten_backward(grad_flat_ptr, batch, self.conv2_out_C, pool2_H, pool2_W)

        # Backprop MaxPool2
        pool2_indices_ptr = self.cache['pool2_indices_ptr']
        grad_conv2_relu_ptr = self.ops.maxpool2d_backward(
            grad_pool2_ptr, pool2_indices_ptr,
            batch, self.conv2_out_C, pool1_H, pool1_W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride,
            0, 0
        )

        # Backprop ReLU (conv2 output) - use pre-ReLU as mask
        conv2_pre_relu_ptr = self.cache['conv2_pre_relu_ptr']
        grad_conv2_ptr = self.ops.alloc(batch * self.conv2_out_C * pool1_H * pool1_W)
        self.ops.relu_backward(grad_conv2_relu_ptr, conv2_pre_relu_ptr, grad_conv2_ptr,
                               batch * self.conv2_out_C * pool1_H * pool1_W)

        # Backprop Conv2 using cuBLAS (need grad_pool1 for further backprop)
        grad_pool1_ptr = self.ops.alloc(batch * self.conv2_in_C * pool1_H * pool1_W)
        pool1_out_ptr = self.cache['pool1_out_ptr']
        self.ops.conv2d_cublas_backward(
            grad_conv2_ptr, pool1_out_ptr, self.conv2_w_ptr,
            batch, self.conv2_in_C, pool1_H, pool1_W,
            self.conv2_out_C, self.conv2_kernel, self.conv2_kernel,
            self.conv2_stride, self.conv2_stride, self.conv2_pad, self.conv2_pad,
            grad_input_ptr=grad_pool1_ptr,
            grad_weight_ptr=self.g_conv2_w_ptr,
            grad_bias_ptr=self.g_conv2_b_ptr,
            col_buffer=self.conv2_col_buffer_ptr,
            grad_col_buffer=self.conv2_grad_col_buffer_ptr,
            grad_gemm_buffer=self.conv2_grad_gemm_buffer_ptr
        )

        # Backprop MaxPool1
        pool1_indices_ptr = self.cache['pool1_indices_ptr']
        grad_conv1_relu_ptr = self.ops.maxpool2d_backward(
            grad_pool1_ptr, pool1_indices_ptr,
            batch, self.conv1_out_C, H, W,
            self.pool_kernel, self.pool_kernel, self.pool_stride, self.pool_stride,
            0, 0
        )

        # Backprop ReLU (conv1 output) - use pre-ReLU as mask
        conv1_pre_relu_ptr = self.cache['conv1_pre_relu_ptr']
        grad_conv1_ptr = self.ops.alloc(batch * self.conv1_out_C * H * W)
        self.ops.relu_backward(grad_conv1_relu_ptr, conv1_pre_relu_ptr, grad_conv1_ptr,
                               batch * self.conv1_out_C * H * W)

        # Backprop Conv1 using cuBLAS (don't need grad_input since input comes from data loader)
        x_ptr = self.cache['x_ptr']
        self.ops.conv2d_cublas_backward(
            grad_conv1_ptr, x_ptr, self.conv1_w_ptr,
            batch, self.conv1_in_C, H, W,
            self.conv1_out_C, self.conv1_kernel, self.conv1_kernel,
            self.conv1_stride, self.conv1_stride, self.conv1_pad, self.conv1_pad,
            grad_input_ptr=None,
            grad_weight_ptr=self.g_conv1_w_ptr,
            grad_bias_ptr=self.g_conv1_b_ptr,
            col_buffer=self.conv1_col_buffer_ptr,
            grad_col_buffer=self.conv1_grad_col_buffer_ptr,
            grad_gemm_buffer=self.conv1_grad_gemm_buffer_ptr
        )

        # Cleanup gradient buffers
        self.ops.free(grad_logits_ptr)
        self.ops.free(grad_flat_ptr)
        self.ops.free(grad_pool2_ptr)
        self.ops.free(grad_conv2_relu_ptr)
        self.ops.free(grad_conv2_ptr)
        self.ops.free(grad_pool1_ptr)
        self.ops.free(grad_conv1_relu_ptr)
        self.ops.free(grad_conv1_ptr)
        self.ops.free(logits_ptr)

        # Cleanup cached activations from forward pass
        for key in ['pool1_indices_ptr', 'pool2_indices_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.lib.cuda_free(self.cache[key])
        for key in ['conv1_pre_relu_ptr', 'conv1_relu_ptr', 'pool1_out_ptr',
                    'conv2_pre_relu_ptr', 'conv2_relu_ptr', 'pool2_out_ptr', 'flat_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.free(self.cache[key])
        self.cache.clear()

        return loss

    def update(self, lr, max_grad_norm=10.0):
        """SGD update on GPU with gradient clipping."""
        # Clip gradients
        self.ops.gradient_clip(self.g_conv1_w_ptr, self.conv1_out_C * self.conv1_in_C * self.conv1_kernel * self.conv1_kernel, max_grad_norm)
        self.ops.gradient_clip(self.g_conv1_b_ptr, self.conv1_out_C, max_grad_norm)
        self.ops.gradient_clip(self.g_conv2_w_ptr, self.conv2_out_C * self.conv2_in_C * self.conv2_kernel * self.conv2_kernel, max_grad_norm)
        self.ops.gradient_clip(self.g_conv2_b_ptr, self.conv2_out_C, max_grad_norm)
        self.ops.gradient_clip(self.g_fc_w_ptr, self.fc_in * self.fc_out, max_grad_norm)
        self.ops.gradient_clip(self.g_fc_b_ptr, self.fc_out, max_grad_norm)

        # SGD update
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
        self.ops.free(logits_ptr)
        # Clear cached activations
        for key in ['pool1_indices_ptr', 'pool2_indices_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.lib.cuda_free(self.cache[key])
        for key in ['conv1_pre_relu_ptr', 'conv1_relu_ptr', 'pool1_out_ptr',
                    'conv2_pre_relu_ptr', 'conv2_relu_ptr', 'pool2_out_ptr', 'flat_ptr']:
            if key in self.cache and self.cache[key]:
                self.ops.free(self.cache[key])
        self.cache.clear()
        return logits.argmax(axis=1)


def test_cnn_model():
    """Test the pure CUDA CNN model with cuBLAS backend."""
    print("Testing SimpleCNN_CUBLAS...")

    ops = CUDAOps()
    model = SimpleCNN_CUBLAS(ops)

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

    print("\n" + "=" * 50)
    print("All cuBLAS CNN model tests passed!")
    print("=" * 50)

    ops.free(x_ptr)


if __name__ == '__main__':
    test_cnn_model()