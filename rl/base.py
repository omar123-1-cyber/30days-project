"""
Day 4: Base Agent & Environment Interfaces
"""

import numpy as np
from typing import Tuple, Any
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """Base class for RL environments."""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (next_state, reward, done, info)."""
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass


class BaseAgent(ABC):
    """Base class for RL agents."""
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Return training metrics."""
        pass
    
    def save(self, filepath: str):
        import json
        params = {f'param_{i}': p.data.tolist() for i, p in enumerate(self.parameters())}
        with open(filepath, 'w') as f:
            json.dump(params, f)
    
    def load(self, filepath: str):
        import json
        with open(filepath, 'r') as f:
            params = json.load(f)
        for i, p in enumerate(self.parameters()):
            p.data = np.array(params[f'param_{i}'], dtype=np.float32)
    
    def parameters(self):
        """Override in subclass."""
        return []