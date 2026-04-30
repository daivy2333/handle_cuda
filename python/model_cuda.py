"""
Pure CUDA MLP Model - All computation on GPU, no numpy matmul in forward/backward
"""

import numpy as np
from cuda_ops import CUDAOps


class SimpleMLP_CUDA:
    """
    Pure CUDA 3-layer MLP for MNIST.
    All forward/backward computation happens on GPU.
    Only weight initialization and data loading use numpy.
    """

    def __init__(self, ops: CUDAOps):
        self.ops = ops
        np.random.seed(42)

        # Initialize weights on CPU (Xavier initialization)
        w1_np = (np.random.randn(784, 256) * np.sqrt(2.0/784)).astype(np.float32)
        b1_np = np.zeros(256, dtype=np.float32)
        w2_np = (np.random.randn(256, 128) * np.sqrt(2.0/256)).astype(np.float32)
        b2_np = np.zeros(128, dtype=np.float32)
        w3_np = (np.random.randn(128, 10) * np.sqrt(2.0/128)).astype(np.float32)
        b3_np = np.zeros(10, dtype=np.float32)

        # Copy weights to GPU (stay on GPU throughout training)
        self.w1_ptr = self.ops.to_device(w1_np)
        self.b1_ptr = self.ops.to_device(b1_np)
        self.w2_ptr = self.ops.to_device(w2_np)
        self.b2_ptr = self.ops.to_device(b2_np)
        self.w3_ptr = self.ops.to_device(w3_np)
        self.b3_ptr = self.ops.to_device(b3_np)

        # Allocate gradient buffers on GPU
        self.gw1_ptr = self.ops.alloc(784 * 256)
        self.gb1_ptr = self.ops.alloc(256)
        self.gw2_ptr = self.ops.alloc(256 * 128)
        self.gb2_ptr = self.ops.alloc(128)
        self.gw3_ptr = self.ops.alloc(128 * 10)
        self.gb3_ptr = self.ops.alloc(10)

        # Cache for backward pass (all GPU pointers)
        self.cache = {}

    def __del__(self):
        """Destructor to free all model-owned GPU allocations."""
        if hasattr(self, 'ops') and self.ops:
            # Free weights
            if hasattr(self, 'w1_ptr'):
                self.ops.free(self.w1_ptr)
            if hasattr(self, 'w2_ptr'):
                self.ops.free(self.w2_ptr)
            if hasattr(self, 'w3_ptr'):
                self.ops.free(self.w3_ptr)
            # Free biases
            if hasattr(self, 'b1_ptr'):
                self.ops.free(self.b1_ptr)
            if hasattr(self, 'b2_ptr'):
                self.ops.free(self.b2_ptr)
            if hasattr(self, 'b3_ptr'):
                self.ops.free(self.b3_ptr)
            # Free gradient buffers
            if hasattr(self, 'gw1_ptr'):
                self.ops.free(self.gw1_ptr)
            if hasattr(self, 'gb1_ptr'):
                self.ops.free(self.gb1_ptr)
            if hasattr(self, 'gw2_ptr'):
                self.ops.free(self.gw2_ptr)
            if hasattr(self, 'gb2_ptr'):
                self.ops.free(self.gb2_ptr)
            if hasattr(self, 'gw3_ptr'):
                self.ops.free(self.gw3_ptr)
            if hasattr(self, 'gb3_ptr'):
                self.ops.free(self.gb3_ptr)
            # Free cached activations if any
            if hasattr(self, 'cache'):
                for key in ['h1_relu_ptr', 'h2_relu_ptr']:
                    if key in self.cache and self.cache[key]:
                        try:
                            self.ops.free(self.cache[key])
                        except:
                            pass

    def forward(self, x_ptr, batch):
        """Pure CUDA forward pass.

        Args:
            x_ptr: GPU pointer to flattened input [batch, 784]
            batch: batch size

        Returns:
            logits_ptr: GPU pointer to logits [batch, 10]
        """
        # Layer 1: matmul + bias + relu
        # x: [batch, 784], w1: [784, 256], h1: [batch, 256]
        h1_ptr = self.ops.matmul(x_ptr, self.w1_ptr, batch, 256, 784)
        h1_b_ptr = self.ops.bias_add(h1_ptr, self.b1_ptr, batch, 256)
        self.ops.relu(h1_b_ptr, batch * 256)  # inplace

        # Layer 2: matmul + bias + relu
        # h1: [batch, 256], w2: [256, 128], h2: [batch, 128]
        h2_ptr = self.ops.matmul(h1_b_ptr, self.w2_ptr, batch, 128, 256)
        h2_b_ptr = self.ops.bias_add(h2_ptr, self.b2_ptr, batch, 128)
        self.ops.relu(h2_b_ptr, batch * 128)  # inplace

        # Layer 3: matmul + bias (no activation)
        # h2: [batch, 128], w3: [128, 10], logits: [batch, 10]
        logits_ptr = self.ops.matmul(h2_b_ptr, self.w3_ptr, batch, 10, 128)
        logits_b_ptr = self.ops.bias_add(logits_ptr, self.b3_ptr, batch, 10)

        # Cache for backward (all GPU pointers)
        self.cache['x_ptr'] = x_ptr
        self.cache['h1_relu_ptr'] = h1_b_ptr  # post-relu output
        self.cache['h2_relu_ptr'] = h2_b_ptr  # post-relu output
        self.cache['batch'] = batch

        # Free intermediate buffers (matmul outputs, not needed for backward)
        self.ops.free(h1_ptr)
        self.ops.free(h2_ptr)
        self.ops.free(logits_ptr)

        return logits_b_ptr

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
        loss, grad_logits_ptr = self.ops.cross_entropy_loss(logits_ptr, targets, batch, 10)

        # Backprop Layer 3: logits = h2_relu @ w3 + b3
        # h2_relu: [batch, 128], w3: [128, 10], logits: [batch, 10]
        h2_relu_ptr = self.cache['h2_relu_ptr']

        # Compute gradients using matmul_backward
        # grad_A (grad_h2_relu) and grad_B (grad_w3)
        grad_h2_relu_ptr, grad_w3_ptr = self.ops.matmul_backward(
            grad_logits_ptr, h2_relu_ptr, self.w3_ptr, batch, 10, 128,
            grad_A_ptr=None, grad_B_ptr=self.gw3_ptr
        )

        # Bias gradient for layer 3
        _, grad_b3_ptr = self.ops.bias_add_backward(grad_logits_ptr, batch, 10,
                                                     grad_input_ptr=None, grad_bias_ptr=self.gb3_ptr)

        # Backprop ReLU layer 2
        # IMPORTANT: relu_backward normally expects the forward input (pre-ReLU), but for ReLU
        # specifically, the mask (input > 0) equals (output > 0) since relu(x) = max(0, x).
        # Therefore, we can pass h2_relu_ptr (post-ReLU output) and it will correctly identify
        # which neurons were active (output > 0) vs inactive (output == 0).
        grad_h2_ptr = self.ops.alloc(batch * 128)
        self.ops.relu_backward(grad_h2_relu_ptr, h2_relu_ptr, grad_h2_ptr, batch * 128)

        # Backprop Layer 2: h2_relu = h1_relu @ w2 + b2
        h1_relu_ptr = self.cache['h1_relu_ptr']

        grad_h1_relu_ptr, grad_w2_ptr = self.ops.matmul_backward(
            grad_h2_ptr, h1_relu_ptr, self.w2_ptr, batch, 128, 256,
            grad_A_ptr=None, grad_B_ptr=self.gw2_ptr
        )

        # Bias gradient for layer 2
        _, grad_b2_ptr = self.ops.bias_add_backward(grad_h2_ptr, batch, 128,
                                                     grad_input_ptr=None, grad_bias_ptr=self.gb2_ptr)

        # Backprop ReLU layer 1
        # IMPORTANT: relu_backward normally expects the forward input (pre-ReLU), but for ReLU
        # specifically, the mask (input > 0) equals (output > 0) since relu(x) = max(0, x).
        # Therefore, we can pass h1_relu_ptr (post-ReLU output) and it will correctly identify
        # which neurons were active (output > 0) vs inactive (output == 0).
        grad_h1_ptr = self.ops.alloc(batch * 256)
        self.ops.relu_backward(grad_h1_relu_ptr, h1_relu_ptr, grad_h1_ptr, batch * 256)

        # Backprop Layer 1: h1_relu = x @ w1 + b1
        x_ptr = self.cache['x_ptr']

        grad_x_ptr, grad_w1_ptr = self.ops.matmul_backward(
            grad_h1_ptr, x_ptr, self.w1_ptr, batch, 256, 784,
            grad_A_ptr=None, grad_B_ptr=self.gw1_ptr
        )

        # Bias gradient for layer 1
        _, grad_b1_ptr = self.ops.bias_add_backward(grad_h1_ptr, batch, 256,
                                                     grad_input_ptr=None, grad_bias_ptr=self.gb1_ptr)

        # Cleanup intermediate buffers
        self.ops.free(grad_logits_ptr)
        self.ops.free(grad_h2_relu_ptr)
        self.ops.free(grad_h2_ptr)
        self.ops.free(grad_h1_relu_ptr)
        self.ops.free(grad_h1_ptr)
        self.ops.free(grad_x_ptr)
        self.ops.free(logits_ptr)

        # Cleanup cached activations from forward pass
        self.cache.clear()

        return loss

    def update(self, lr):
        """SGD update on GPU."""
        self.ops.sgd_update(self.w1_ptr, self.gw1_ptr, 784 * 256, lr)
        self.ops.sgd_update(self.b1_ptr, self.gb1_ptr, 256, lr)
        self.ops.sgd_update(self.w2_ptr, self.gw2_ptr, 256 * 128, lr)
        self.ops.sgd_update(self.b2_ptr, self.gb2_ptr, 128, lr)
        self.ops.sgd_update(self.w3_ptr, self.gw3_ptr, 128 * 10, lr)
        self.ops.sgd_update(self.b3_ptr, self.gb3_ptr, 10, lr)

    def predict(self, x_ptr, batch):
        """Predict on GPU, return numpy array."""
        logits_ptr = self.forward(x_ptr, batch)
        logits = self.ops.to_host(logits_ptr, (batch, 10))
        # Free logits buffer
        self.ops.free(logits_ptr)
        # Clear cached activations (inference doesn't need backward)
        self.cache.clear()
        return logits.argmax(axis=1)


def test_model_cuda():
    """Test the pure CUDA MLP model."""
    print("Testing SimpleMLP_CUDA...")

    ops = CUDAOps()
    model = SimpleMLP_CUDA(ops)

    # Test forward
    print("\n1. Testing forward pass...")
    batch = 32
    x = np.random.randn(batch, 784).astype(np.float32)
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
    # Need to create new forward pass since previous logits were freed
    preds = model.predict(x_ptr, batch)
    assert preds.shape == (batch,)
    assert np.all(preds >= 0) and np.all(preds < 10)
    print(f"   Predictions: {preds[:5]}")
    print("   Predict: PASSED")

    # Test full training iteration
    print("\n5. Testing full training iteration...")
    x2 = np.random.randn(batch, 784).astype(np.float32)
    x2_ptr = ops.to_device(x2)
    targets2 = np.random.randint(0, 10, batch).astype(np.int32)

    logits2_ptr = model.forward(x2_ptr, batch)
    loss2 = model.backward(logits2_ptr, targets2)
    model.update(lr=0.01)
    print(f"   Loss after update: {loss2:.4f}")
    assert not np.isnan(loss2)
    print("   Full iteration: PASSED")

    ops.free(x_ptr)
    ops.free(x2_ptr)

    print("\n" + "="*50)
    print("All CUDA model tests passed!")
    print("="*50)


class SimpleMLP_MixedPrecision:
    """
    Mixed Precision MLP: FP32 master weights + FP16 working weights.

    Pattern:
    - FP32 master weights (w_fp32) for SGD update (numerical stability)
    - FP16 working weights (w_fp16) for matmul (Tensor Core acceleration)
    - FP16 activations for matmul inputs
    - FP32 for bias_add, relu, cross_entropy (simple ops, no tensor core benefit)
    - Loss scaling for gradient underflow prevention
    """

    def __init__(self, ops: CUDAOps, loss_scale=128.0):
        self.ops = ops
        self.loss_scale = loss_scale
        np.random.seed(42)

        # Initialize FP32 master weights
        w1_np = (np.random.randn(784, 256) * np.sqrt(2.0/784)).astype(np.float32)
        b1_np = np.zeros(256, dtype=np.float32)
        w2_np = (np.random.randn(256, 128) * np.sqrt(2.0/256)).astype(np.float32)
        b2_np = np.zeros(128, dtype=np.float32)
        w3_np = (np.random.randn(128, 10) * np.sqrt(2.0/128)).astype(np.float32)
        b3_np = np.zeros(10, dtype=np.float32)

        # FP32 master weights on GPU
        self.w1_fp32 = ops.to_device(w1_np)
        self.b1_fp32 = ops.to_device(b1_np)
        self.w2_fp32 = ops.to_device(w2_np)
        self.b2_fp32 = ops.to_device(b2_np)
        self.w3_fp32 = ops.to_device(w3_np)
        self.b3_fp32 = ops.to_device(b3_np)

        # FP16 working weights on GPU
        self.w1_fp16 = ops.alloc_fp16(784 * 256)
        self.w2_fp16 = ops.alloc_fp16(256 * 128)
        self.w3_fp16 = ops.alloc_fp16(128 * 10)

        # Convert master to working
        ops.float_to_half(self.w1_fp32, self.w1_fp16, 784 * 256)
        ops.float_to_half(self.w2_fp32, self.w2_fp16, 256 * 128)
        ops.float_to_half(self.w3_fp32, self.w3_fp16, 128 * 10)

        # FP32 gradient buffers
        self.gw1_fp32 = ops.alloc(784 * 256)
        self.gb1_fp32 = ops.alloc(256)
        self.gw2_fp32 = ops.alloc(256 * 128)
        self.gb2_fp32 = ops.alloc(128)
        self.gw3_fp32 = ops.alloc(128 * 10)
        self.gb3_fp32 = ops.alloc(10)

        # Cache for backward
        self.cache = {}

    def __del__(self):
        if hasattr(self, 'ops') and self.ops:
            for attr in ['w1_fp32', 'w2_fp32', 'w3_fp32',
                         'b1_fp32', 'b2_fp32', 'b3_fp32',
                         'w1_fp16', 'w2_fp16', 'w3_fp16',
                         'gw1_fp32', 'gw2_fp32', 'gw3_fp32',
                         'gb1_fp32', 'gb2_fp32', 'gb3_fp32']:
                if hasattr(self, attr):
                    self.ops.free(getattr(self, attr))

    def forward(self, x_fp32_ptr, batch):
        """Mixed precision forward pass.

        Args:
            x_fp32_ptr: GPU pointer to FP32 input [batch, 784]
            batch: batch size

        Returns:
            logits_ptr: GPU pointer to FP32 logits [batch, 10]
        """
        # Convert input to FP16
        x_fp16_ptr = self.ops.alloc_fp16(batch * 784)
        self.ops.float_to_half(x_fp32_ptr, x_fp16_ptr, batch * 784)

        # Layer 1: FP16 matmul -> FP32 bias+relu
        h1_fp32_ptr = self.ops.matmul_fp16(x_fp16_ptr, self.w1_fp16, batch, 256, 784)
        h1_b_fp32_ptr = self.ops.bias_add(h1_fp32_ptr, self.b1_fp32, batch, 256)
        self.ops.relu(h1_b_fp32_ptr, batch * 256)  # inplace

        # Convert to FP16 for next matmul
        h1_fp16_ptr = self.ops.alloc_fp16(batch * 256)
        self.ops.float_to_half(h1_b_fp32_ptr, h1_fp16_ptr, batch * 256)

        # Layer 2: FP16 matmul -> FP32 bias+relu
        h2_fp32_ptr = self.ops.matmul_fp16(h1_fp16_ptr, self.w2_fp16, batch, 128, 256)
        h2_b_fp32_ptr = self.ops.bias_add(h2_fp32_ptr, self.b2_fp32, batch, 128)
        self.ops.relu(h2_b_fp32_ptr, batch * 128)  # inplace

        # Convert to FP16 for next matmul
        h2_fp16_ptr = self.ops.alloc_fp16(batch * 128)
        self.ops.float_to_half(h2_b_fp32_ptr, h2_fp16_ptr, batch * 128)

        # Layer 3: FP16 matmul -> FP32 bias
        logits_fp32_ptr = self.ops.matmul_fp16(h2_fp16_ptr, self.w3_fp16, batch, 10, 128)
        logits_b_fp32_ptr = self.ops.bias_add(logits_fp32_ptr, self.b3_fp32, batch, 10)

        # Cache for backward
        self.cache = {
            'x_fp16': x_fp16_ptr,
            'h1_fp16': h1_fp16_ptr,
            'h2_fp16': h2_fp16_ptr,
            'h1_relu_fp32': h1_b_fp32_ptr,
            'h2_relu_fp32': h2_b_fp32_ptr,
            'batch': batch
        }

        # Free intermediate FP32 matmul outputs
        self.ops.free(h1_fp32_ptr)
        self.ops.free(h2_fp32_ptr)
        self.ops.free(logits_fp32_ptr)

        return logits_b_fp32_ptr

    def backward(self, logits_fp32_ptr, targets):
        """Mixed precision backward pass with loss scaling.

        Args:
            logits_fp32_ptr: GPU pointer to FP32 logits
            targets: numpy int32 array [batch]

        Returns:
            loss: scalar loss value
        """
        batch = self.cache['batch']

        # Cross entropy loss (FP32)
        loss, grad_logits_fp32_ptr = self.ops.cross_entropy_loss(logits_fp32_ptr, targets, batch, 10)

        # Apply loss scaling
        self.ops.scale_gradients(grad_logits_fp32_ptr, batch * 10, self.loss_scale)

        # Backprop Layer 3: FP16 backward
        h2_fp16_ptr = self.cache['h2_fp16']

        grad_h2_fp32_ptr, grad_w3_fp32_ptr = self.ops.matmul_fp16_backward(
            grad_logits_fp32_ptr, h2_fp16_ptr, self.w3_fp16,
            batch, 10, 128,
            grad_B_fp32_ptr=self.gw3_fp32
        )

        # Unscale weight gradients
        self.ops.scale_gradients(self.gw3_fp32, 128 * 10, 1.0 / self.loss_scale)

        # Bias backward (FP32)
        _, grad_b3_fp32_ptr = self.ops.bias_add_backward(
            grad_logits_fp32_ptr, batch, 10,
            grad_bias_ptr=self.gb3_fp32
        )

        # Unscale bias gradient
        self.ops.scale_gradients(self.gb3_fp32, 10, 1.0 / self.loss_scale)

        # ReLU backward (FP32)
        h2_relu_fp32_ptr = self.cache['h2_relu_fp32']
        grad_h2_relu_fp32_ptr = self.ops.alloc(batch * 128)
        self.ops.relu_backward(grad_h2_fp32_ptr, h2_relu_fp32_ptr, grad_h2_relu_fp32_ptr, batch * 128)

        # Unscale and convert for next layer
        self.ops.scale_gradients(grad_h2_relu_fp32_ptr, batch * 128, 1.0 / self.loss_scale)
        grad_h2_fp16_ptr = self.ops.alloc_fp16(batch * 128)
        self.ops.float_to_half(grad_h2_relu_fp32_ptr, grad_h2_fp16_ptr, batch * 128)

        # Backprop Layer 2: FP16 backward
        h1_fp16_ptr = self.cache['h1_fp16']

        grad_h1_fp32_ptr, grad_w2_fp32_ptr = self.ops.matmul_fp16_backward(
            grad_h2_fp16_ptr, h1_fp16_ptr, self.w2_fp16,
            batch, 128, 256,
            grad_B_fp32_ptr=self.gw2_fp32
        )

        self.ops.scale_gradients(self.gw2_fp32, 256 * 128, 1.0 / self.loss_scale)

        _, grad_b2_fp32_ptr = self.ops.bias_add_backward(
            grad_h2_relu_fp32_ptr, batch, 128,
            grad_bias_ptr=self.gb2_fp32
        )

        self.ops.scale_gradients(self.gb2_fp32, 128, 1.0 / self.loss_scale)

        # ReLU backward Layer 1
        h1_relu_fp32_ptr = self.cache['h1_relu_fp32']
        grad_h1_relu_fp32_ptr = self.ops.alloc(batch * 256)
        self.ops.relu_backward(grad_h1_fp32_ptr, h1_relu_fp32_ptr, grad_h1_relu_fp32_ptr, batch * 256)

        self.ops.scale_gradients(grad_h1_relu_fp32_ptr, batch * 256, 1.0 / self.loss_scale)
        grad_h1_fp16_ptr = self.ops.alloc_fp16(batch * 256)
        self.ops.float_to_half(grad_h1_relu_fp32_ptr, grad_h1_fp16_ptr, batch * 256)

        # Backprop Layer 1: FP16 backward
        x_fp16_ptr = self.cache['x_fp16']

        grad_x_fp32_ptr, grad_w1_fp32_ptr = self.ops.matmul_fp16_backward(
            grad_h1_fp16_ptr, x_fp16_ptr, self.w1_fp16,
            batch, 256, 784,
            grad_B_fp32_ptr=self.gw1_fp32
        )

        self.ops.scale_gradients(self.gw1_fp32, 784 * 256, 1.0 / self.loss_scale)

        _, grad_b1_fp32_ptr = self.ops.bias_add_backward(
            grad_h1_relu_fp32_ptr, batch, 256,
            grad_bias_ptr=self.gb1_fp32
        )

        self.ops.scale_gradients(self.gb1_fp32, 256, 1.0 / self.loss_scale)

        # Cleanup - free all allocated buffers including cached ones
        # Free gradient buffers
        self.ops.free(grad_logits_fp32_ptr)
        self.ops.free(grad_h2_fp32_ptr)
        self.ops.free(grad_h2_relu_fp32_ptr)
        self.ops.free(grad_h2_fp16_ptr)
        self.ops.free(grad_h1_fp32_ptr)
        self.ops.free(grad_h1_relu_fp32_ptr)
        self.ops.free(grad_h1_fp16_ptr)
        self.ops.free(grad_x_fp32_ptr)
        # Free logits (returned by forward, not freed yet)
        self.ops.free(logits_fp32_ptr)

        # Free cached FP16 buffers from forward pass
        if 'x_fp16' in self.cache:
            self.ops.free(self.cache['x_fp16'])
        if 'h1_fp16' in self.cache:
            self.ops.free(self.cache['h1_fp16'])
        if 'h2_fp16' in self.cache:
            self.ops.free(self.cache['h2_fp16'])
        if 'h1_relu_fp32' in self.cache:
            self.ops.free(self.cache['h1_relu_fp32'])
        if 'h2_relu_fp32' in self.cache:
            self.ops.free(self.cache['h2_relu_fp32'])

        self.cache.clear()

        return loss

    def update(self, lr):
        """SGD update on FP32 master weights, sync to FP16 working."""
        # Update FP32 masters
        self.ops.sgd_update(self.w1_fp32, self.gw1_fp32, 784 * 256, lr)
        self.ops.sgd_update(self.b1_fp32, self.gb1_fp32, 256, lr)
        self.ops.sgd_update(self.w2_fp32, self.gw2_fp32, 256 * 128, lr)
        self.ops.sgd_update(self.b2_fp32, self.gb2_fp32, 128, lr)
        self.ops.sgd_update(self.w3_fp32, self.gw3_fp32, 128 * 10, lr)
        self.ops.sgd_update(self.b3_fp32, self.gb3_fp32, 10, lr)

        # Sync FP16 working weights from FP32 masters
        self.ops.float_to_half(self.w1_fp32, self.w1_fp16, 784 * 256)
        self.ops.float_to_half(self.w2_fp32, self.w2_fp16, 256 * 128)
        self.ops.float_to_half(self.w3_fp32, self.w3_fp16, 128 * 10)

    def predict(self, x_fp32_ptr, batch):
        """Predict on GPU."""
        logits_ptr = self.forward(x_fp32_ptr, batch)
        logits = self.ops.to_host(logits_ptr, (batch, 10))
        self.ops.free(logits_ptr)
        self.cache.clear()
        return logits.argmax(axis=1)


def test_model_mixed_precision():
    """Test the mixed precision MLP model."""
    print("Testing SimpleMLP_MixedPrecision...")

    ops = CUDAOps()
    model = SimpleMLP_MixedPrecision(ops, loss_scale=128.0)

    # Test forward
    print("\n1. Testing forward pass...")
    batch = 32
    x = np.random.randn(batch, 784).astype(np.float32)
    x_ptr = ops.to_device(x)
    logits_ptr = model.forward(x_ptr, batch)
    logits = ops.to_host(logits_ptr, (batch, 10))
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (batch, 10)
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
    preds = model.predict(x_ptr, batch)
    assert preds.shape == (batch,)
    print(f"   Predictions: {preds[:5]}")
    print("   Predict: PASSED")

    ops.free(x_ptr)

    print("\n" + "="*50)
    print("Mixed precision model tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_model_cuda()
    print("\n")
    test_model_mixed_precision()
