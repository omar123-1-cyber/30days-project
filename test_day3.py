"""
Day 3 Test: MNIST Digit Classification from Scratch!
"""

import numpy as np
from core.tensor import Tensor
from core.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from core.optim import Adam
from core.data import DataLoader, train_test_split, normalize
from core.utils import Trainer, accuracy, save_model, save_history

# ==================== Generate Toy MNIST-like Data ====================
# Real MNIST er bodole synthetic data diye test korbo
# Real MNIST chailo: sklearn.datasets.load_digits use korte paro

print("=== Generating synthetic MNIST-like data ===")
np.random.seed(42)

n_samples = 5000
n_features = 784  # 28x28
n_classes = 10

# Create synthetic digits with some structure
X = np.random.randn(n_samples, n_features).astype(np.float32) * 0.5
y = np.random.randint(0, n_classes, size=n_samples)

# Add some class-dependent structure (simple pattern)
for i in range(n_samples):
    class_pattern = np.sin(np.linspace(0, 2 * np.pi * (y[i] + 1), n_features)) * 0.3
    X[i] += class_pattern

# Normalize
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, random_seed=42)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {np.unique(y_train)}")

# ==================== Model ====================
print("\n=== Building Model ===")
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)  # 10 classes
)

print(f"Total parameters: {sum(p.data.size for p in model.parameters())}")

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# ==================== Training ====================
print("\n=== Training Start ===")
train_loader = DataLoader(X_train, y_train, batch_size=64, shuffle=True)
val_loader = DataLoader(X_test, y_test, batch_size=128, shuffle=False)

trainer = Trainer(model, loss_fn, optimizer, metrics=['accuracy'])
history = trainer.fit(train_loader, val_loader, epochs=30, verbose=1)

# ==================== Final Evaluation ====================
print("\n=== Final Test Evaluation ===")
test_loss, test_acc = trainer.evaluate(val_loader)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# Predictions
preds = trainer.predict(X_test)
acc = accuracy(preds, y_test)
print(f"Manual Accuracy Check: {acc:.4f}")

# ==================== Save ====================
print("\n=== Saving Model ===")
save_model(model, 'mnist_model.npz')
save_history(history, 'training_history.json')

print("\n✅ Day 3 complete! Full MNIST pipeline works!")
print("Next: Day 4 - Replay Buffer & RL Environment Base")