#!/usr/bin/env python3
"""
Compare our CUDA implementation with PyTorch
"""

import sys
sys.path.insert(0, '/home/daivy/projects/handle_cuda/.claude/worktrees/cnn-mnist-training/python')

import numpy as np
import torch
import torch.nn as nn
import time
from cuda_ops import CUDAOps
from mnist_data import load_mnist
from model import SimpleMLP

def create_torch_model():
    """PyTorch equivalent of our SimpleMLP"""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def test_correctness():
    """Test operator correctness"""
    print("=== Correctness Tests ===")

    ops = CUDAOps()

    print("\n1. CrossEntropyLoss:")
    batch, classes = 32, 10
    np.random.seed(42)
    logits_np = np.random.randn(batch, classes).astype(np.float32)
    targets_np = np.random.randint(0, classes, batch).astype(np.int32)

    # Our implementation
    logits_ptr = ops.to_device(logits_np)
    our_loss, grad_ptr = ops.cross_entropy_loss(logits_ptr, targets_np, batch, classes)
    our_grad = ops.to_host(grad_ptr, (batch, classes))

    # PyTorch
    torch_logits = torch.from_numpy(logits_np.copy())
    torch_targets = torch.from_numpy(targets_np).long()
    criterion = nn.CrossEntropyLoss()
    torch_logits.requires_grad_(True)
    torch_loss = criterion(torch_logits, torch_targets)
    torch_loss.backward()
    torch_grad = torch_logits.grad.numpy()

    print(f"  Loss: ours={our_loss:.4f}, torch={torch_loss.item():.4f}, diff={abs(our_loss-torch_loss.item()):.6f}")
    print(f"  Grad max diff: {np.abs(our_grad - torch_grad).max():.6f}")

    if abs(our_loss - torch_loss.item()) < 1e-4:
        print("  PASSED")

def test_convergence():
    """Compare training convergence"""
    print("\n=== Convergence Comparison (3 epochs) ===")

    train_images, train_labels = load_mnist(train=True)

    ops = CUDAOps()
    our_model = SimpleMLP(ops)

    torch_model = create_torch_model()
    torch_model.train()
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    n_batches = 100

    for epoch in range(3):
        our_correct = 0
        torch_correct = 0

        for i in range(n_batches):
            images = train_images[i*batch_size:(i+1)*batch_size]
            labels = train_labels[i*batch_size:(i+1)*batch_size]

            # Our model
            logits = our_model.forward(images)
            our_model.backward(logits, labels)
            our_model.update(0.1)
            our_correct += (logits.argmax(axis=1) == labels).sum()

            # PyTorch
            x = torch.from_numpy(images.reshape(batch_size, 784))
            y = torch.from_numpy(labels).long()
            optimizer.zero_grad()
            output = torch_model(x)
            criterion(output, y).backward()
            optimizer.step()
            torch_correct += (output.argmax(axis=1).numpy() == labels).sum()

        print(f"Epoch {epoch+1}: ours={our_correct/(n_batches*batch_size):.2%}, torch={torch_correct/(n_batches*batch_size):.2%}")

def test_performance():
    """Compare training speed"""
    print("\n=== Performance Comparison ===")

    train_images, train_labels = load_mnist(train=True)
    batch_size = 64
    n_batches = 50

    ops = CUDAOps()
    our_model = SimpleMLP(ops)

    start = time.time()
    for i in range(n_batches):
        images = train_images[i*batch_size:(i+1)*batch_size]
        labels = train_labels[i*batch_size:(i+1)*batch_size]
        logits = our_model.forward(images)
        our_model.backward(logits, labels)
        our_model.update(0.1)
    our_time = time.time() - start

    torch_model = create_torch_model()
    torch_model.train()
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for i in range(n_batches):
        images = train_images[i*batch_size:(i+1)*batch_size].reshape(batch_size, 784)
        labels = train_labels[i*batch_size:(i+1)*batch_size]
        x = torch.from_numpy(images)
        y = torch.from_numpy(labels).long()
        optimizer.zero_grad()
        criterion(torch_model(x), y).backward()
        optimizer.step()
    torch_time = time.time() - start

    print(f"  Our: {our_time:.2f}s ({our_time/n_batches*1000:.1f}ms/batch)")
    print(f"  PyTorch: {torch_time:.2f}s ({torch_time/n_batches*1000:.1f}ms/batch)")
    print(f"  Ratio: {our_time/torch_time:.2f}x")

if __name__ == '__main__':
    test_correctness()
    test_convergence()
    test_performance()
    print("\nAll comparisons complete")