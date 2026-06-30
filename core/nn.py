"""
NeuroTopology - Day 2/3: Neural Network Layers & Loss Functions
"""

import numpy as np
from typing import List
from .tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self) -> List[Tensor]:
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
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Fully connected layer: y = x @ W + b"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
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


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        # Numerical stability
        shifted = x.data - np.max(x.data, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        return Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        self.mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        out = Tensor(x.data * self.mask / (1 - self.p), requires_grad=x.requires_grad, _children=(x,), _op="dropout")
        
        def _backward():
            if x.requires_grad:
                grad = out.grad * self.mask / (1 - self.p)
                x.grad = x.grad + grad if x.grad is not None else grad
        out._backward = _backward
        return out


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
    
    def train(self, mode: bool = True):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = mode
        return self
    
    def eval(self):
        return self.train(False)


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
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        
        log_softmax = shifted - np.log(np.sum(exp_shifted, axis=1, keepdims=True))
        
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