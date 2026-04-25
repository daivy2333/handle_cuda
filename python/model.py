"""
Simple MLP Model for MNIST (simplified version for testing)
"""

import numpy as np
from cuda_ops import CUDAOps

class SimpleMLP:
    """
    Simple 3-layer MLP for MNIST:
    - Input: 784 (flattened 28x28)
    - Hidden1: 256 -> ReLU
    - Hidden2: 128 -> ReLU
    - Output: 10
    """

    def __init__(self, ops: CUDAOps):
        self.ops = ops
        np.random.seed(42)

        # Initialize weights (Xavier initialization)
        self.w1 = (np.random.randn(784, 256) * np.sqrt(2.0/784)).astype(np.float32)
        self.b1 = np.zeros(256, dtype=np.float32)

        self.w2 = (np.random.randn(256, 128) * np.sqrt(2.0/256)).astype(np.float32)
        self.b2 = np.zeros(128, dtype=np.float32)

        self.w3 = (np.random.randn(128, 10) * np.sqrt(2.0/128)).astype(np.float32)
        self.b3 = np.zeros(10, dtype=np.float32)

        # Allocate CUDA memory
        self.dw1 = self.ops.to_device(self.w1)
        self.db1 = self.ops.to_device(self.b1)
        self.dw2 = self.ops.to_device(self.w2)
        self.db2 = self.ops.to_device(self.b2)
        self.dw3 = self.ops.to_device(self.w3)
        self.db3 = self.ops.to_device(self.b3)

        # Gradient buffers
        self.gw1 = self.ops.alloc(784 * 256)
        self.gb1 = self.ops.alloc(256)
        self.gw2 = self.ops.alloc(256 * 128)
        self.gb2 = self.ops.alloc(128)
        self.gw3 = self.ops.alloc(128 * 10)
        self.gb3 = self.ops.alloc(10)

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray):
        """Forward pass - simplified using numpy matmul for now"""
        batch = x.shape[0]

        # Flatten input
        x_flat = x.reshape(batch, 784)

        # Layer 1: Linear + ReLU
        h1 = x_flat @ self.w1 + self.b1
        h1_relu = np.maximum(0, h1)

        # Layer 2: Linear + ReLU
        h2 = h1_relu @ self.w2 + self.b2
        h2_relu = np.maximum(0, h2)

        # Layer 3: Linear
        logits = h2_relu @ self.w3 + self.b3

        # Cache for backward
        self.cache['x_flat'] = x_flat
        self.cache['h1'] = h1
        self.cache['h1_relu'] = h1_relu
        self.cache['h2'] = h2
        self.cache['h2_relu'] = h2_relu

        return logits

    def backward(self, logits: np.ndarray, targets: np.ndarray):
        """Backward pass using numpy"""
        batch = logits.shape[0]

        # Cross entropy gradient (from CUDA)
        logits_ptr = self.ops.to_device(logits)
        loss, grad_logits_ptr = self.ops.cross_entropy_loss(logits_ptr, targets, batch, 10)
        grad_logits = self.ops.to_host(grad_logits_ptr, (batch, 10))

        # Backprop through layer 3
        h2_relu = self.cache['h2_relu']
        self.gw3_np = h2_relu.T @ grad_logits  # [128, 10]
        self.gb3_np = grad_logits.sum(axis=0)  # [10]
        grad_h2 = grad_logits @ self.w3.T  # [batch, 128]

        # Backprop through ReLU layer 2
        grad_h2_relu = grad_h2 * (self.cache['h2'] > 0).astype(np.float32)

        # Backprop through layer 2
        h1_relu = self.cache['h1_relu']
        self.gw2_np = h1_relu.T @ grad_h2_relu  # [256, 128]
        self.gb2_np = grad_h2_relu.sum(axis=0)  # [128]
        grad_h1 = grad_h2_relu @ self.w2.T  # [batch, 256]

        # Backprop through ReLU layer 1
        grad_h1_relu = grad_h1 * (self.cache['h1'] > 0).astype(np.float32)

        # Backprop through layer 1
        x_flat = self.cache['x_flat']
        self.gw1_np = x_flat.T @ grad_h1_relu  # [784, 256]
        self.gb1_np = grad_h1_relu.sum(axis=0)  # [256]

        # Copy gradients to device
        self.ops.lib.cuda_memcpy_h2d(self.gw1, self.gw1_np.ctypes.data, self.gw1_np.nbytes)
        self.ops.lib.cuda_memcpy_h2d(self.gb1, self.gb1_np.ctypes.data, self.gb1_np.nbytes)
        self.ops.lib.cuda_memcpy_h2d(self.gw2, self.gw2_np.ctypes.data, self.gw2_np.nbytes)
        self.ops.lib.cuda_memcpy_h2d(self.gb2, self.gb2_np.ctypes.data, self.gb2_np.nbytes)
        self.ops.lib.cuda_memcpy_h2d(self.gw3, self.gw3_np.ctypes.data, self.gw3_np.nbytes)
        self.ops.lib.cuda_memcpy_h2d(self.gb3, self.gb3_np.ctypes.data, self.gb3_np.nbytes)

        self.ops.free(logits_ptr)
        self.ops.free(grad_logits_ptr)

        return loss

    def update(self, lr: float):
        """SGD update"""
        self.ops.sgd_update(self.dw1, self.gw1, 784 * 256, lr)
        self.ops.sgd_update(self.db1, self.gb1, 256, lr)
        self.ops.sgd_update(self.dw2, self.gw2, 256 * 128, lr)
        self.ops.sgd_update(self.db2, self.gb2, 128, lr)
        self.ops.sgd_update(self.dw3, self.gw3, 128 * 10, lr)
        self.ops.sgd_update(self.db3, self.gb3, 10, lr)

        # Sync weights to host
        self.w1 = self.ops.to_host(self.dw1, (784, 256))
        self.b1 = self.ops.to_host(self.db1, (256,))
        self.w2 = self.ops.to_host(self.dw2, (256, 128))
        self.b2 = self.ops.to_host(self.db2, (128,))
        self.w3 = self.ops.to_host(self.dw3, (128, 10))
        self.b3 = self.ops.to_host(self.db3, (10,))

    def predict(self, x: np.ndarray):
        """Predict class labels"""
        logits = self.forward(x)
        return logits.argmax(axis=1)


def test_model():
    """Test the SimpleMLP model."""
    print("Testing SimpleMLP model...")

    ops = CUDAOps()
    model = SimpleMLP(ops)

    # Test forward pass
    print("\n1. Testing forward pass...")
    batch_size = 32
    x = np.random.randn(batch_size, 28, 28).astype(np.float32)
    logits = model.forward(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 10), f"Expected shape (32, 10), got {logits.shape}"
    print("   Forward pass: PASSED")

    # Test backward pass
    print("\n2. Testing backward pass...")
    targets = np.random.randint(0, 10, batch_size).astype(np.int32)
    loss = model.backward(logits, targets)
    print(f"   Loss: {loss:.4f}")
    assert not np.isnan(loss), "Loss is NaN!"
    assert hasattr(model, 'gw1_np'), "Gradients not computed!"
    assert model.gw1_np.shape == (784, 256), f"gw1 shape mismatch: {model.gw1_np.shape}"
    assert model.gw2_np.shape == (256, 128), f"gw2 shape mismatch: {model.gw2_np.shape}"
    assert model.gw3_np.shape == (128, 10), f"gw3 shape mismatch: {model.gw3_np.shape}"
    print("   Backward pass: PASSED")

    # Test update
    print("\n3. Testing SGD update...")
    old_w1 = model.w1.copy()
    old_b1 = model.b1.copy()
    model.update(lr=0.01)
    # Weights should change
    assert not np.allclose(model.w1, old_w1), "Weights not updated!"
    assert not np.allclose(model.b1, old_b1), "Biases not updated!"
    print("   SGD update: PASSED")

    # Test predict
    print("\n4. Testing predict...")
    predictions = model.predict(x)
    assert predictions.shape == (batch_size,), f"Expected shape (32,), got {predictions.shape}"
    assert np.all(predictions >= 0) and np.all(predictions < 10), "Predictions out of range!"
    print(f"   Predictions: {predictions[:5]}...")
    print("   Predict: PASSED")

    print("\n" + "="*50)
    print("All model tests passed!")
    print("="*50)


if __name__ == '__main__':
    test_model()