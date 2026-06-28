"""
NeuroTopology - Day 2: Optimizers
SGD with Momentum and Adam.
"""

import numpy as np
from typing import List
from .tensor import Tensor


class Optimizer:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()
    
    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = {id(p): np.zeros_like(p.data) for p in self.parameters}
    
    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue
            if self.momentum > 0:
                self.velocity[id(p)] = self.momentum * self.velocity[id(p)] + p.grad
                p.data = p.data - self.lr * self.velocity[id(p)]
            else:
                p.data = p.data - self.lr * p.grad


class Adam(Optimizer):
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        
        self.m = {id(p): np.zeros_like(p.data) for p in self.parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in self.parameters}
    
    def step(self) -> None:
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue
            
            # Biased first moment
            self.m[id(p)] = self.betas[0] * self.m[id(p)] + (1 - self.betas[0]) * p.grad
            # Biased second moment
            self.v[id(p)] = self.betas[1] * self.v[id(p)] + (1 - self.betas[1]) * (p.grad ** 2)
            
            # Bias correction
            m_hat = self.m[id(p)] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[id(p)] / (1 - self.betas[1] ** self.t)
            
            # Update
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)