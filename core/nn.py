"""
NeuroTopology - Day 2: Neural Network Layers & Loss Functions
Tensor engine er upor built ML layers.
"""

import numpy as np
from typing import List
from .tensor import Tensor


class Module:
    """Base class for all neural network modules. (PyTorch style)"""
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params
    
    def zero_grad(self) -> None:
        """Zero gradients of all parameters."""
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Fully connected layer: y = x @ W + b"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )
        self.b = Tensor(
            np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MSELoss(Module):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """
    CrossEntropyLoss = LogSoftmax + NLLLoss
    y_true: integer class indices, shape (batch,)
    logits: raw model output, shape (batch, num_classes)
    """
    def forward(self, logits: Tensor, y_true: np.ndarray) -> Tensor:
        # Numerical stability: shift by max
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        
        # Log-softmax
        log_softmax = shifted - np.log(np.sum(exp_shifted, axis=1, keepdims=True))
        
        # Negative log likelihood
        batch_size = logits.data.shape[0]
        correct_log_probs = log_softmax[np.arange(batch_size), y_true]
        loss_val = -np.mean(correct_log_probs)
        
        out = Tensor(loss_val, requires_grad=True, _children=(logits,), _op="crossentropy")
        
        def _backward():
            if logits.requires_grad:
                grad = softmax.copy()
                grad[np.arange(batch_size), y_true] -= 1
                grad = grad / batch_size
                logits.grad = logits.grad + grad if logits.grad is not None else grad
        
        out._backward = _backward
        return out