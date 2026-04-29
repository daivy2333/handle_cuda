"""
PyTorch CNN Benchmark for comparison with Pure CUDA implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from mnist_data import load_mnist


class PyTorchCNN(nn.Module):
    """Standard PyTorch CNN matching our CUDA architecture."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

        # Xavier initialization (same as CUDA model)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_pytorch():
    """Train PyTorch CNN on MNIST."""
    print("=" * 60)
    print("  PyTorch CNN Training on MNIST (GPU)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n[1/3] Loading MNIST dataset...")
    start_load = time.time()
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)
    load_time = time.time() - start_load
    print(f"      Loaded in {load_time:.2f}s")

    # Convert to PyTorch tensors
    train_x = torch.from_numpy(train_images).to(device)
    train_y = torch.from_numpy(train_labels.astype(np.int64)).to(device)
    test_x = torch.from_numpy(test_images).to(device)
    test_y = torch.from_numpy(test_labels.astype(np.int64)).to(device)

    # Model setup
    print("\n[2/3] Initializing model...")
    model = PyTorchCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    epochs = 10
    num_batches = train_images.shape[0] // batch_size

    print(f"      Batch size: {batch_size}")
    print(f"      Learning rate: 0.01")
    print(f"      Epochs: {epochs}")

    history = {'loss': [], 'acc': [], 'epoch_time': []}

    print("\n[3/3] Training...")
    print("-" * 60)
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_times = []

        model.train()
        for i in range(num_batches):
            batch_start = time.time()

            x_batch = train_x[i*batch_size:(i+1)*batch_size]
            y_batch = train_y[i*batch_size:(i+1)*batch_size]

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if (i + 1) % 100 == 0 or i == 0:
                avg_batch_time = np.mean(batch_times[-100:]) if len(batch_times) >= 100 else np.mean(batch_times)
                samples_per_sec = batch_size / avg_batch_time

                progress_pct = (i + 1) / num_batches * 100
                bar_len = 20
                filled = int(bar_len * (i + 1) / num_batches)
                bar = '█' * filled + '░' * (bar_len - filled)

                print(f"  Epoch {epoch+1}/{epochs} [{bar}] "
                      f"{i+1:4d}/{num_batches} | "
                      f"Loss: {epoch_loss/(i+1):.4f} | "
                      f"Speed: {samples_per_sec:6.0f} samples/s", flush=True)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_batch_time = np.mean(batch_times)
        samples_per_sec = batch_size / avg_batch_time

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_output = model(test_x)
            preds = test_output.argmax(dim=1)
            test_acc = (preds == test_y).float().mean().item()

        history['loss'].append(avg_loss)
        history['acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        elapsed_total = time.time() - total_start
        epochs_left = epochs - epoch - 1
        eta_total = epochs_left * epoch_time

        print(f"\n  ✓ Epoch {epoch+1} complete in {epoch_time:.1f}s")
        print(f"    Avg loss: {avg_loss:.4f} | Test accuracy: {test_acc:.2%} | "
              f"Speed: {samples_per_sec:.0f} samples/s")
        print(f"    Total elapsed: {elapsed_total:.1f}s | "
              f"ETA for remaining {epochs_left} epochs: {eta_total:.1f}s")
        print("-" * 60)

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg epoch time: {np.mean(history['epoch_time']):.1f}s")
    print(f"  Avg samples/sec: {train_images.shape[0] / np.mean(history['epoch_time']):.0f}")
    print(f"  Final test accuracy: {history['acc'][-1]:.2%}")
    print(f"  Loss trajectory: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}")
    print("=" * 60)

    return history, total_time


if __name__ == '__main__':
    train_pytorch()