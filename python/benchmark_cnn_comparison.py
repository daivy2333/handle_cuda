"""
MNIST Training Comparison: CUDA cuBLAS vs PyTorch
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/home/daivy/projects/handle_cuda/python')

from cuda_ops import CUDAOps
from model_cnn_cublas import SimpleCNN_CUBLAS
from mnist_data import load_mnist


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


def train_cuda_cnn():
    """Train CUDA CNN with cuBLAS backend."""
    print("=" * 60)
    print("  Pure CUDA CNN Training (cuBLAS Backend)")
    print("=" * 60)

    print("\nLoading MNIST dataset...")
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)
    print(f"Train: {train_images.shape[0]} samples")
    print(f"Test:  {test_images.shape[0]} samples")

    ops = CUDAOps()
    model = SimpleCNN_CUBLAS(ops, batch_size=64)

    batch_size = 64
    lr = 0.01
    epochs = 5
    num_batches = train_images.shape[0] // batch_size

    print(f"\nConfig: batch={batch_size}, lr={lr}, epochs={epochs}")
    print(f"Batches per epoch: {num_batches}")

    # Pre-allocate batch buffer
    x_batch_ptr = ops.alloc(batch_size * 1 * 28 * 28)

    history = {'loss': [], 'acc': [], 'epoch_time': []}

    print("\nTraining...")
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

            # Progress
            if (i + 1) % 100 == 0 or i == num_batches - 1:
                avg_batch_time = np.mean(batch_times[-100:])
                samples_per_sec = batch_size / avg_batch_time
                print(f"  Epoch {epoch+1}/{epochs} Batch {i+1:4d}/{num_batches} | "
                      f"Loss: {epoch_loss/(i+1):.4f} | "
                      f"Speed: {samples_per_sec:6.0f} samples/s", flush=True)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_batch_time = np.mean(batch_times)
        samples_per_sec = batch_size / avg_batch_time

        # Evaluate
        test_acc = evaluate_cuda(model, ops, test_images, test_labels)

        history['loss'].append(avg_loss)
        history['acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        print(f"\n  Epoch {epoch+1}: time={format_time(epoch_time)}, "
              f"loss={avg_loss:.4f}, acc={test_acc:.2%}, speed={samples_per_sec:.0f} samples/s")
        print("-" * 60)

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("  CUDA Training Complete!")
    print("=" * 60)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Avg epoch time: {format_time(np.mean(history['epoch_time']))}")
    print(f"  Avg samples/sec: {train_images.shape[0] / np.mean(history['epoch_time']):.0f}")
    print(f"  Final accuracy: {history['acc'][-1]:.2%}")
    print("=" * 60)

    ops.free(x_batch_ptr)
    return history, total_time


def evaluate_cuda(model, ops, images, labels, batch_size=1000):
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


def train_pytorch_cnn():
    """Train PyTorch CNN for comparison."""
    print("\n" + "=" * 60)
    print("  PyTorch CNN Training (Reference)")
    print("=" * 60)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Same architecture as CUDA model
    class PyTorchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('/tmp/mnist', train=False, transform=transform)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = PyTorchCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    history = {'loss': [], 'acc': [], 'epoch_time': []}

    print(f"\nConfig: batch={batch_size}, lr=0.01, epochs={epochs}")
    print("\nTraining...")
    print("-" * 60)

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_times = []
        num_batches = len(train_loader)

        model.train()
        for i, (data, target) in enumerate(train_loader):
            batch_start = time.time()

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if (i + 1) % 100 == 0 or i == num_batches - 1:
                avg_batch_time = np.mean(batch_times[-100:])
                samples_per_sec = batch_size / avg_batch_time
                print(f"  Epoch {epoch+1}/{epochs} Batch {i+1:4d}/{num_batches} | "
                      f"Loss: {epoch_loss/(i+1):.4f} | "
                      f"Speed: {samples_per_sec:6.0f} samples/s", flush=True)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        test_acc = correct / total
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        samples_per_sec = batch_size / np.mean(batch_times)

        history['loss'].append(avg_loss)
        history['acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        print(f"\n  Epoch {epoch+1}: time={format_time(epoch_time)}, "
              f"loss={avg_loss:.4f}, acc={test_acc:.2%}, speed={samples_per_sec:.0f} samples/s")
        print("-" * 60)

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("  PyTorch Training Complete!")
    print("=" * 60)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Avg epoch time: {format_time(np.mean(history['epoch_time']))}")
    print(f"  Avg samples/sec: {60000 / np.mean(history['epoch_time']):.0f}")
    print(f"  Final accuracy: {history['acc'][-1]:.2%}")
    print("=" * 60)

    return history, total_time


def main():
    """Run comparison."""
    print("\n" + "=" * 70)
    print("  MNIST CNN Training Comparison: CUDA cuBLAS vs PyTorch")
    print("=" * 70)

    # Train CUDA model
    cuda_history, cuda_time = train_cuda_cnn()

    # Train PyTorch model
    pytorch_history, pytorch_time = train_pytorch_cnn()

    # Comparison summary
    print("\n" + "=" * 70)
    print("  Performance Comparison Summary")
    print("=" * 70)

    cuda_speed = 60000 / np.mean(cuda_history['epoch_time'])
    pytorch_speed = 60000 / np.mean(pytorch_history['epoch_time'])

    print(f"\n  CUDA cuBLAS:")
    print(f"    Total time: {format_time(cuda_time)}")
    print(f"    Avg samples/sec: {cuda_speed:.0f}")
    print(f"    Final accuracy: {cuda_history['acc'][-1]:.2%}")

    print(f"\n  PyTorch:")
    print(f"    Total time: {format_time(pytorch_time)}")
    print(f"    Avg samples/sec: {pytorch_speed:.0f}")
    print(f"    Final accuracy: {pytorch_history['acc'][-1]:.2%}")

    print(f"\n  Speedup:")
    ratio = cuda_speed / pytorch_speed
    if ratio < 1:
        print(f"    PyTorch is {1/ratio:.1f}x faster than CUDA")
    else:
        print(f"    CUDA is {ratio:.1f}x faster than PyTorch")

    print(f"\n  Gap closed: CUDA achieves {ratio*100:.1f}% of PyTorch speed")
    print("=" * 70)


if __name__ == '__main__':
    main()