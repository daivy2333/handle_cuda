"""
Mixed Precision MLP with Pre-allocated Buffers - Fixed Version
"""

import numpy as np
import ctypes

class SimpleMLP_MixedPrecision_Fixed:
    """
    Mixed Precision MLP with pre-allocated buffers to avoid memory issues.

    Pattern:
    - FP32 master weights for SGD update
    - FP16 working weights for matmul
    - Pre-allocated buffers to avoid repeated alloc/free
    """

    def __init__(self, ops, batch_size=64, loss_scale=128.0):
        self.ops = ops
        self.batch_size = batch_size
        self.loss_scale = loss_scale

        # Dimensions
        self.d_in = 784
        self.d_h1 = 256
        self.d_h2 = 128
        self.d_out = 10

        np.random.seed(42)

        # FP32 master weights
        w1 = (np.random.randn(self.d_in, self.d_h1) * np.sqrt(2.0/self.d_in)).astype(np.float32)
        b1 = np.zeros(self.d_h1, dtype=np.float32)
        w2 = (np.random.randn(self.d_h1, self.d_h2) * np.sqrt(2.0/self.d_h1)).astype(np.float32)
        b2 = np.zeros(self.d_h2, dtype=np.float32)
        w3 = (np.random.randn(self.d_h2, self.d_out) * np.sqrt(2.0/self.d_h2)).astype(np.float32)
        b3 = np.zeros(self.d_out, dtype=np.float32)

        self.w1_fp32 = ops.to_device(w1)
        self.b1_fp32 = ops.to_device(b1)
        self.w2_fp32 = ops.to_device(w2)
        self.b2_fp32 = ops.to_device(b2)
        self.w3_fp32 = ops.to_device(w3)
        self.b3_fp32 = ops.to_device(b3)

        # FP16 working weights
        self.w1_fp16 = ops.alloc_fp16(self.d_in * self.d_h1)
        self.w2_fp16 = ops.alloc_fp16(self.d_h1 * self.d_h2)
        self.w3_fp16 = ops.alloc_fp16(self.d_h2 * self.d_out)

        ops.float_to_half(self.w1_fp32, self.w1_fp16, self.d_in * self.d_h1)
        ops.float_to_half(self.w2_fp32, self.w2_fp16, self.d_h1 * self.d_h2)
        ops.float_to_half(self.w3_fp32, self.w3_fp16, self.d_h2 * self.d_out)

        # FP32 gradient buffers
        self.gw1_fp32 = ops.alloc(self.d_in * self.d_h1)
        self.gb1_fp32 = ops.alloc(self.d_h1)
        self.gw2_fp32 = ops.alloc(self.d_h1 * self.d_h2)
        self.gb2_fp32 = ops.alloc(self.d_h2)
        self.gw3_fp32 = ops.alloc(self.d_h2 * self.d_out)
        self.gb3_fp32 = ops.alloc(self.d_out)

        # Pre-allocated buffers (batch_size)
        b = batch_size
        self.buf_x_fp16 = ops.alloc_fp16(b * self.d_in)
        self.buf_h1_fp16 = ops.alloc_fp16(b * self.d_h1)
        self.buf_h2_fp16 = ops.alloc_fp16(b * self.d_h2)
        self.buf_h1_relu = ops.alloc(b * self.d_h1)
        self.buf_h2_relu = ops.alloc(b * self.d_h2)
        self.buf_logits_bias = ops.alloc(b * self.d_out)

        # Backward buffers
        self.buf_grad_logits = ops.alloc(b * self.d_out)
        self.buf_grad_h2 = ops.alloc(b * self.d_h2)
        self.buf_grad_h2_relu = ops.alloc(b * self.d_h2)
        self.buf_grad_h2_fp16 = ops.alloc_fp16(b * self.d_h2)
        self.buf_grad_h1 = ops.alloc(b * self.d_h1)
        self.buf_grad_h1_relu = ops.alloc(b * self.d_h1)
        self.buf_grad_h1_fp16 = ops.alloc_fp16(b * self.d_h1)

        # Matmul output buffers (need to free each forward)
        self.buf_h1_out = None
        self.buf_h2_out = None
        self.buf_logits_out = None

    def forward(self, x_fp32_ptr, batch):
        """Forward pass using pre-allocated buffers."""
        assert batch <= self.batch_size, f"batch {batch} > max {self.batch_size}"

        # Convert input to FP16
        self.ops.float_to_half(x_fp32_ptr, self.buf_x_fp16, batch * self.d_in)

        # Layer 1: matmul -> bias -> relu
        self.buf_h1_out = self.ops.matmul_fp16(self.buf_x_fp16, self.w1_fp16, batch, self.d_h1, self.d_in)
        self.ops.bias_add_inplace(self.buf_h1_out, self.b1_fp32, batch, self.d_h1)
        self.ops.relu(self.buf_h1_out, batch * self.d_h1)
        self.ops.float_to_half(self.buf_h1_out, self.buf_h1_fp16, batch * self.d_h1)

        # Layer 2: matmul -> bias -> relu
        self.buf_h2_out = self.ops.matmul_fp16(self.buf_h1_fp16, self.w2_fp16, batch, self.d_h2, self.d_h1)
        self.ops.bias_add_inplace(self.buf_h2_out, self.b2_fp32, batch, self.d_h2)
        self.ops.relu(self.buf_h2_out, batch * self.d_h2)
        self.ops.float_to_half(self.buf_h2_out, self.buf_h2_fp16, batch * self.d_h2)

        # Layer 3: matmul -> bias
        self.buf_logits_out = self.ops.matmul_fp16(self.buf_h2_fp16, self.w3_fp16, batch, self.d_out, self.d_h2)
        self.ops.bias_add_outplace(self.buf_logits_out, self.b3_fp32, self.buf_logits_bias, batch, self.d_out)

        self.batch = batch
        return self.buf_logits_bias

    def backward(self, logits_ptr, targets):
        """Backward pass with loss scaling."""
        batch = self.batch

        # Cross entropy loss
        loss, grad_logits = self.ops.cross_entropy_loss(logits_ptr, targets, batch, self.d_out)

        # Scale gradients
        self.ops.scale_gradients(grad_logits, batch * self.d_out, self.loss_scale)

        # Layer 3 backward
        self.ops.matmul_fp16_backward(
            grad_logits, self.buf_h2_fp16, self.w3_fp16,
            batch, self.d_out, self.d_h2,
            grad_A_fp32_ptr=self.buf_grad_h2,
            grad_B_fp32_ptr=self.gw3_fp32
        )
        self.ops.scale_gradients(self.gw3_fp32, self.d_h2 * self.d_out, 1.0 / self.loss_scale)

        self.ops.bias_add_backward(grad_logits, batch, self.d_out, grad_bias_ptr=self.gb3_fp32)
        self.ops.scale_gradients(self.gb3_fp32, self.d_out, 1.0 / self.loss_scale)

        # ReLU backward Layer 2
        self.ops.relu_backward(self.buf_grad_h2, self.buf_h2_out, self.buf_grad_h2_relu, batch * self.d_h2)
        self.ops.scale_gradients(self.buf_grad_h2_relu, batch * self.d_h2, 1.0 / self.loss_scale)
        self.ops.float_to_half(self.buf_grad_h2_relu, self.buf_grad_h2_fp16, batch * self.d_h2)

        # Layer 2 backward
        self.ops.matmul_fp16_backward(
            self.buf_grad_h2_fp16, self.buf_h1_fp16, self.w2_fp16,
            batch, self.d_h2, self.d_h1,
            grad_A_fp32_ptr=self.buf_grad_h1,
            grad_B_fp32_ptr=self.gw2_fp32
        )
        self.ops.scale_gradients(self.gw2_fp32, self.d_h1 * self.d_h2, 1.0 / self.loss_scale)

        self.ops.bias_add_backward(self.buf_grad_h2_relu, batch, self.d_h2, grad_bias_ptr=self.gb2_fp32)
        self.ops.scale_gradients(self.gb2_fp32, self.d_h2, 1.0 / self.loss_scale)

        # ReLU backward Layer 1
        self.ops.relu_backward(self.buf_grad_h1, self.buf_h1_out, self.buf_grad_h1_relu, batch * self.d_h1)
        self.ops.scale_gradients(self.buf_grad_h1_relu, batch * self.d_h1, 1.0 / self.loss_scale)
        self.ops.float_to_half(self.buf_grad_h1_relu, self.buf_grad_h1_fp16, batch * self.d_h1)

        # Layer 1 backward
        self.ops.matmul_fp16_backward(
            self.buf_grad_h1_fp16, self.buf_x_fp16, self.w1_fp16,
            batch, self.d_h1, self.d_in,
            grad_A_fp32_ptr=None,  # Don't need grad_x
            grad_B_fp32_ptr=self.gw1_fp32
        )
        self.ops.scale_gradients(self.gw1_fp32, self.d_in * self.d_h1, 1.0 / self.loss_scale)

        self.ops.bias_add_backward(self.buf_grad_h1_relu, batch, self.d_h1, grad_bias_ptr=self.gb1_fp32)
        self.ops.scale_gradients(self.gb1_fp32, self.d_h1, 1.0 / self.loss_scale)

        # Free matmul output buffers
        if self.buf_h1_out:
            self.ops.free(self.buf_h1_out)
            self.buf_h1_out = None
        if self.buf_h2_out:
            self.ops.free(self.buf_h2_out)
            self.buf_h2_out = None
        if self.buf_logits_out:
            self.ops.free(self.buf_logits_out)
            self.buf_logits_out = None
        self.ops.free(grad_logits)

        return loss

    def update(self, lr):
        """SGD update on FP32 master weights."""
        self.ops.sgd_update(self.w1_fp32, self.gw1_fp32, self.d_in * self.d_h1, lr)
        self.ops.sgd_update(self.b1_fp32, self.gb1_fp32, self.d_h1, lr)
        self.ops.sgd_update(self.w2_fp32, self.gw2_fp32, self.d_h1 * self.d_h2, lr)
        self.ops.sgd_update(self.b2_fp32, self.gb2_fp32, self.d_h2, lr)
        self.ops.sgd_update(self.w3_fp32, self.gw3_fp32, self.d_h2 * self.d_out, lr)
        self.ops.sgd_update(self.b3_fp32, self.gb3_fp32, self.d_out, lr)

        # Sync FP16 working weights
        self.ops.float_to_half(self.w1_fp32, self.w1_fp16, self.d_in * self.d_h1)
        self.ops.float_to_half(self.w2_fp32, self.w2_fp16, self.d_h1 * self.d_h2)
        self.ops.float_to_half(self.w3_fp32, self.w3_fp16, self.d_h2 * self.d_out)

    def __del__(self):
        if hasattr(self, 'ops') and self.ops:
            for attr in ['w1_fp32', 'w2_fp32', 'w3_fp32',
                         'b1_fp32', 'b2_fp32', 'b3_fp32',
                         'w1_fp16', 'w2_fp16', 'w3_fp16',
                         'gw1_fp32', 'gw2_fp32', 'gw3_fp32',
                         'gb1_fp32', 'gb2_fp32', 'gb3_fp32',
                         'buf_x_fp16', 'buf_h1_fp16', 'buf_h2_fp16',
                         'buf_h1_relu', 'buf_h2_relu', 'buf_logits_bias',
                         'buf_grad_logits', 'buf_grad_h2', 'buf_grad_h2_relu',
                         'buf_grad_h2_fp16', 'buf_grad_h1', 'buf_grad_h1_relu',
                         'buf_grad_h1_fp16']:
                if hasattr(self, attr):
                    try:
                        self.ops.free(getattr(self, attr))
                    except:
                        pass