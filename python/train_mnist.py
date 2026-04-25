#!/usr/bin/env python3
"""
MNIST Training Script
"""

import sys
sys.path.insert(0, '/home/daivy/projects/handle_cuda/.claude/worktrees/cnn-mnist-training/python')

import numpy as np
import time
from cuda_ops import CUDAOps
from mnist_data import load_mnist, get_batches
from model import SimpleMLP

def train(model, ops, train_images, train_labels, epochs=5, batch_size=64, lr=0.01):
    n_batches = len(train_images) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(get_batches(train_images, train_labels, batch_size)):
            # Forward
            logits = model.forward(images)

            # Backward + Loss
            loss = model.backward(logits, labels)
            epoch_loss += loss

            # Update
            model.update(lr)

            # Accuracy
            predictions = logits.argmax(axis=1)
            correct += (predictions == labels).sum()
            total += len(labels)

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{n_batches}: loss={loss:.4f}")

        epoch_time = time.time() - start_time
        train_acc = correct / total
        avg_loss = epoch_loss / n_batches

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={train_acc:.2%}, time={epoch_time:.2f}s")

def evaluate(model, test_images, test_labels, batch_size=100):
    correct = 0
    total = 0

    for images, labels in get_batches(test_images, test_labels, batch_size, shuffle=False):
        logits = model.forward(images)
        predictions = logits.argmax(axis=1)
        correct += (predictions == labels).sum()
        total += len(labels)

    return correct / total

def main():
    print("Loading MNIST...")
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)

    print("Initializing CUDA...")
    ops = CUDAOps()

    print("Creating model...")
    model = SimpleMLP(ops)

    print("\nTraining...")
    train(model, ops, train_images, train_labels, epochs=5, batch_size=64, lr=0.1)

    print("\nEvaluating...")
    test_acc = evaluate(model, test_images, test_labels)
    print(f"Test accuracy: {test_acc:.2%}")

if __name__ == '__main__':
    main()