"""
MNIST Dataset Loader
"""

import numpy as np
import gzip
import os
import urllib.request

MNIST_URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}

def download_mnist(data_dir='data/mnist'):
    os.makedirs(data_dir, exist_ok=True)
    for name, url in MNIST_URLS.items():
        filepath = os.path.join(data_dir, name + '.gz')
        if not os.path.exists(filepath):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
    print("Download complete!")

def load_mnist(data_dir='data/mnist', train=True):
    download_mnist(data_dir)

    if train:
        images_path = os.path.join(data_dir, 'train_images.gz')
        labels_path = os.path.join(data_dir, 'train_labels.gz')
    else:
        images_path = os.path.join(data_dir, 'test_images.gz')
        labels_path = os.path.join(data_dir, 'test_labels.gz')

    # Load images
    with gzip.open(images_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    images = data.reshape(-1, 28, 28).astype(np.float32) / 255.0

    # Normalize same as PyTorch (mean=0.1307, std=0.3081)
    images = (images - 0.1307) / 0.3081

    images = images[:, np.newaxis, :, :]  # [N, 1, 28, 28]

    # Load labels
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return images, labels.astype(np.int32)

def get_batches(images, labels, batch_size, shuffle=True):
    n = len(images)
    if shuffle:
        indices = np.random.permutation(n)
        images = images[indices]
        labels = labels[indices]

    for i in range(0, n, batch_size):
        yield images[i:i+batch_size], labels[i:i+batch_size]

if __name__ == '__main__':
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)

    print(f"Train: {train_images.shape}, {train_labels.shape}")
    print(f"Test: {test_images.shape}, {test_labels.shape}")
    print(f"Label range: {train_labels.min()} to {train_labels.max()}")