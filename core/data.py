"""
NeuroTopology - Day 3: Data Loading Utilities
"""

import numpy as np
from typing import Tuple, Iterator


class DataLoader:
    """Simple batch data loader."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
    
    def __len__(self) -> int:
        return self.n_batches


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, random_seed: int = 42) -> Tuple:
    """Split data into train and test sets."""
    np.random.seed(random_seed)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_ratio))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> Tuple:
    """Z-score normalization."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    n = len(y)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), y.astype(int)] = 1
    return one_hot


def load_mnist_from_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST from CSV format.
    First column = label, rest = pixel values (0-255)
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    y = data[:, 0].astype(np.int32)
    X = data[:, 1:].astype(np.float32) / 255.0  # Normalize to [0, 1]
    return X, y