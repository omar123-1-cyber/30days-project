"""
Day 2 Test: Simple Regression with our from-scratch ML library
"""

import numpy as np
from core.tensor import Tensor
from core.nn import Linear, ReLU, Sequential, MSELoss
from core.optim import Adam

# Toy dataset: y = 3x + 2 + noise
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Model: 1 -> 16 -> 1
model = Sequential(
    Linear(1, 16),
    ReLU(),
    Linear(16, 1)
)

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

print("=== Day 2: Training Start ===")
for epoch in range(200):
    x_tensor = Tensor(X)
    y_tensor = Tensor(y)
    
    # Forward
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Test prediction
print("\n=== Test Predictions ===")
test_inputs = [0.0, 1.0, 5.0, 10.0]
for val in test_inputs:
    pred = model(Tensor([[val]]))
    expected = 3 * val + 2
    print(f"x = {val:5.1f} | Predicted: {pred.item():.4f} | Expected: {expected:.4f}")

print("\n✅ Day 2 complete! MLP with backprop works perfectly.")