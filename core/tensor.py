"""
NeuroTopology - Day 1: Custom Tensor Engine with Autograd
NumPy diye from scratch reverse-mode automatic differentiation.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Union


class Tensor:
    """
    NumPy array er upor built Tensor class.
    Reverse-mode autograd support kore.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, list, float],
        dtype: np.dtype = np.float32,
        requires_grad: bool = False,
        _children: Tuple = (),
        _op: str = ""
    ):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable = lambda: None
        self._prev = set(_children)
        self._op = _op
        
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=dtype)
    
    # ==================== Utility Methods ====================
    
    def zero_grad(self) -> None:
        """Gradient zero kore."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)
    
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def T(self) -> "Tensor":
        """Transpose."""
        return self.transpose()
    
    def item(self) -> float:
        """Single element tensor ke Python float e convert."""
        return float(self.data.item())
    
    def __repr__(self) -> str:
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}, dtype={self.data.dtype}{grad_str})"
    
    def __len__(self) -> int:
        return len(self.data)
    
    # ==================== Core Operations ====================
    
    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add"
        )
        
        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                # Broadcasting handle korar jonno sum kore correct shape e anbo
                if self.data.shape != out.grad.shape:
                    grad_self = self._unbroadcast(out.grad, self.data.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            
            if other.requires_grad:
                grad_other = out.grad
                if other.data.shape != out.grad.shape:
                    grad_other = self._unbroadcast(out.grad, other.data.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Union[float, int]) -> "Tensor":
        return self.__add__(other)
    
    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.__add__(-other)
    
    def __rsub__(self, other: Union[float, int]) -> "Tensor":
        return (-self).__add__(other)
    
    def __neg__(self) -> "Tensor":
        return self * (-1)
    
    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul"
        )
        
        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                if self.data.shape != grad_self.shape:
                    grad_self = self._unbroadcast(grad_self, self.data.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            
            if other.requires_grad:
                grad_other = out.grad * self.data
                if other.data.shape != grad_other.shape:
                    grad_other = self._unbroadcast(grad_other, other.data.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Union[float, int]) -> "Tensor":
        return self.__mul__(other)
    
    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)
    
    def __rtruediv__(self, other: Union[float, int]) -> "Tensor":
        return (self ** -1) * other
    
    def __pow__(self, other: Union[float, int]) -> "Tensor":
        assert isinstance(other, (int, float)), "Power must be scalar"
        out = Tensor(
            self.data ** other,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow{other}"
        )
        
        def _backward():
            if self.requires_grad:
                grad = other * (self.data ** (other - 1)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul"
        )
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (out.grad @ other.data.T) if self.grad is not None else (out.grad @ other.data.T)
            if other.requires_grad:
                other.grad = other.grad + (self.data.T @ out.grad) if other.grad is not None else (self.data.T @ out.grad)
        
        out._backward = _backward
        return out
    
    # ==================== Activation Functions ====================
    
    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu"
        )
        
        def _backward():
            if self.requires_grad:
                grad = (self.data > 0).astype(self.data.dtype) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> "Tensor":
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid"
        )
        
        def _backward():
            if self.requires_grad:
                grad = sig * (1 - sig) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh"
        )
        
        def _backward():
            if self.requires_grad:
                grad = (1 - t ** 2) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def exp(self) -> "Tensor":
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.data * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data + 1e-8),  # numerical stability
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log"
        )
        
        def _backward():
            if self.requires_grad:
                grad = (1 / (self.data + 1e-8)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    # ==================== Reduction Ops ====================
    
    def sum(self, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum"
        )
        
        def _backward():
            if self.requires_grad:
                grad = np.broadcast_to(out.grad, self.data.shape) if not keepdims else out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Optional[Union[int, Tuple]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean"
        )
        
        def _backward():
            if self.requires_grad:
                n = self.data.size if axis is None else self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])
                grad = np.broadcast_to(out.grad / n, self.data.shape) if not keepdims else out.grad / n
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="max"
        )
        
        def _backward():
            if self.requires_grad:
                if axis is None:
                    mask = (self.data == out_data).astype(self.data.dtype)
                    grad = mask * out.grad / np.sum(mask)
                else:
                    if not keepdims:
                        out_grad_exp = np.expand_dims(out.grad, axis=axis)
                    else:
                        out_grad_exp = out.grad
                    mask = (self.data == out_data).astype(self.data.dtype)
                    grad = mask * out_grad_exp / np.sum(mask, axis=axis, keepdims=True)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    # ==================== Shape Ops ====================
    
    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def transpose(self, axes: Optional[Tuple] = None) -> "Tensor":
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose"
        )
        
        def _backward():
            if self.requires_grad:
                if axes is None:
                    grad = out.grad.T
                else:
                    inv_axes = tuple(np.argsort(axes))
                    grad = out.grad.transpose(inv_axes)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    # ==================== Backward Pass ====================
    
    def backward(self) -> None:
        """
        Topological sort diye computation graph traverse kore
        reverse-mode autograd chalay.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() call korar jonno requires_grad=True lagbe")
        
        # Seed gradient
        if self.grad is None or self.grad.shape == ():
            self.grad = np.ones_like(self.data)
        
        # Topological sort
        topo: List[Tensor] = []
        visited = set()
        
        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Reverse order e backward chalao
        for node in reversed(topo):
            node._backward()
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def _unbroadcast(grad: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """
        Broadcasting er jonno gradient ke original shape e reduce kore.
        """
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)
        
        for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
            if g_dim != t_dim and t_dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
    
    def numpy(self) -> np.ndarray:
        """NumPy array return kore (copy)."""
        return self.data.copy()
    
    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod
    def ones(*shape: int, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape: int, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)
    
    @staticmethod
    def rand(*shape: int, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)
    
    @staticmethod
    def arange(start: int, stop: int, step: int = 1, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.arange(start, stop, step, dtype=np.float32), requires_grad=requires_grad)
    
    @staticmethod
    def linspace(start: float, stop: float, num: int, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.linspace(start, stop, num, dtype=np.float32), requires_grad=requires_grad)