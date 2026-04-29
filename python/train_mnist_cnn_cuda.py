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

    # Training config
    batch_size = 64
    lr = 0.01
    epochs = 10

    # Pre-allocate batch buffer (CNN input is [batch, 1, 28, 28])
    x_batch_ptr = ops.alloc(batch_size * 1 * 28 * 28)

    history = {'loss': [], 'acc': []}

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = train_images.shape[0] // batch_size

        for i in range(num_batches):
            # Get batch (CPU) - no reshape needed, data is already [N, 1, 28, 28]
            x_batch = train_images[i*batch_size:(i+1)*batch_size]
            y_batch = train_labels[i*batch_size:(i+1)*batch_size]

            # Copy to GPU
            ops.lib.cuda_memcpy_h2d(x_batch_ptr, x_batch.ctypes.data, x_batch.nbytes)

            # Forward (GPU)
            logits_ptr = model.forward(x_batch_ptr, batch_size)

            # Backward (GPU, targets stay on CPU)
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

    ops.free(x_batch_ptr)

    return history, total_time


def evaluate_cnn(model, ops, images, labels, batch_size=1000):
    """Evaluate accuracy on test set."""
    correct = 0
    total = images.shape[0]

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        actual_batch = end - i

        # No reshape needed - data is already [N, 1, 28, 28]
        x = images[i:end]
        x_ptr = ops.to_device(x)

        preds = model.predict(x_ptr, actual_batch)
        correct += np.sum(preds == labels[i:end])

        ops.free(x_ptr)

    return correct / total


if __name__ == '__main__':
    train_cnn()