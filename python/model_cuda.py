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


if __name__ == '__main__':
    test_model_cuda()
