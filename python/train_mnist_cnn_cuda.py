"""
MNIST Training with Pure CUDA CNN - Enhanced Progress Display
"""

import numpy as np
import time
from cuda_ops import CUDAOps
from model_cnn_cuda import SimpleCNN_CUDA
from mnist_data import load_mnist


def train_cnn():
    """Train pure CUDA CNN on MNIST with detailed progress."""
    print("=" * 60)
    print("  Pure CUDA CNN Training on MNIST")
    print("=" * 60)

    print("\n[1/3] Loading MNIST dataset...")
    start_load = time.time()
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)
    load_time = time.time() - start_load
    print(f"      Loaded in {load_time:.2f}s")
    print(f"      Train: {train_images.shape[0]} samples")
    print(f"      Test:  {test_images.shape[0]} samples")

    print("\n[2/3] Initializing model...")
    ops = CUDAOps()
    model = SimpleCNN_CUDA(ops)

    # Training config
    batch_size = 64
    lr = 0.01
    epochs = 10
    num_batches = train_images.shape[0] // batch_size
    progress_interval = 50  # Print progress every N batches

    print(f"      Batch size: {batch_size}")
    print(f"      Learning rate: {lr}")
    print(f"      Epochs: {epochs}")
    print(f"      Batches per epoch: {num_batches}")
    print(f"      Progress interval: every {progress_interval} batches")
    print(f"      Expected time per batch: ~0.10s (forward 0.05s + backward 0.05s)")
    print(f"      Expected epoch time: ~{num_batches * 0.10 / 60:.1f} min")

    # Pre-allocate batch buffer
    x_batch_ptr = ops.alloc(batch_size * 1 * 28 * 28)

    history = {'loss': [], 'acc': [], 'epoch_time': []}

    print("\n[3/3] Training...")
    print("-" * 60)
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_times = []

        for i in range(num_batches):
            batch_start = time.time()

            # Get batch
            x_batch = train_images[i*batch_size:(i+1)*batch_size]
            y_batch = train_labels[i*batch_size:(i+1)*batch_size]

            # Copy to GPU
            ops.lib.cuda_memcpy_h2d(x_batch_ptr, x_batch.ctypes.data, x_batch.nbytes)

            # Forward, backward, update
            logits_ptr = model.forward(x_batch_ptr, batch_size)
            loss = model.backward(logits_ptr, y_batch)
            model.update(lr)

            epoch_loss += loss
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Progress output - every batch for first epoch, then every 50
            progress_freq = 1 if epoch == 0 else 50
            if (i + 1) % progress_freq == 0 or i == 0 or i == num_batches - 1:
                avg_batch_time = np.mean(batch_times[-100:]) if len(batch_times) >= 100 else np.mean(batch_times)
                samples_per_sec = batch_size / avg_batch_time
                batches_done = i + 1
                batches_left = num_batches - batches_done
                eta_sec = batches_left * avg_batch_time

                # Progress bar
                progress_pct = batches_done / num_batches * 100
                bar_len = 20
                filled = int(bar_len * batches_done / num_batches)
                bar = '█' * filled + '░' * (bar_len - filled)

                print(f"  Epoch {epoch+1}/{epochs} [{bar}] "
                      f"{batches_done:4d}/{num_batches} | "
                      f"Loss: {epoch_loss/batches_done:.4f} | "
                      f"Speed: {samples_per_sec:6.0f} samples/s | "
                      f"ETA: {format_time(eta_sec)}", flush=True)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_batch_time = np.mean(batch_times)
        samples_per_sec = batch_size / avg_batch_time

        # Evaluate
        test_acc = evaluate_cnn(model, ops, test_images, test_labels)

        history['loss'].append(avg_loss)
        history['acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        elapsed_total = time.time() - total_start
        epochs_left = epochs - epoch - 1
        eta_total = epochs_left * epoch_time

        print(f"\n  ✓ Epoch {epoch+1} complete in {format_time(epoch_time)}")
        print(f"    Avg loss: {avg_loss:.4f} | Test accuracy: {test_acc:.2%} | "
              f"Speed: {samples_per_sec:.0f} samples/s")
        print(f"    Total elapsed: {format_time(elapsed_total)} | "
              f"ETA for remaining {epochs_left} epochs: {format_time(eta_total)}")
        print("-" * 60)

    total_time = time.time() - total_start

    # Final summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Avg epoch time: {format_time(np.mean(history['epoch_time']))}")
    print(f"  Avg samples/sec: {train_images.shape[0] / np.mean(history['epoch_time']):.0f}")
    print(f"  Final test accuracy: {history['acc'][-1]:.2%}")
    print(f"  Loss trajectory: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}")
    print(f"  Accuracy trajectory: {history['acc'][0]:.2%} → {history['acc'][-1]:.2%}")
    print("=" * 60)

    ops.free(x_batch_ptr)

    return history, total_time


def evaluate_cnn(model, ops, images, labels, batch_size=1000):
    """Evaluate accuracy on test set."""
    correct = 0
    total = images.shape[0]

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        actual_batch = end - i

        x = images[i:end]
        x_ptr = ops.to_device(x)

        preds = model.predict(x_ptr, actual_batch)
        correct += np.sum(preds == labels[i:end])

        ops.free(x_ptr)

    return correct / total


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


if __name__ == '__main__':
    train_cnn()