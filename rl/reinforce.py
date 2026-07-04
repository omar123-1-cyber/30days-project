"""
Day 8: REINFORCE — Monte Carlo Policy Gradient
"""

import numpy as np
from core.tensor import Tensor
from core.nn import Module, Linear, ReLU, Sequential
from core.optim import Adam
from .base import BaseAgent


class PolicyNetwork(Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.net = Sequential(
            Linear(state_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        logits = self.net(x)
        # Softmax for action probabilities
        shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        return Tensor(probs, requires_grad=True, _children=(logits,), _op='softmax')


class REINFORCEAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_t = Tensor(state.reshape(1, -1))
        probs = self.policy(state_t).data[0]
        
        if training:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = np.argmax(probs)
        return int(action)
    
    def update(self, state, action, reward, next_state, done) -> dict:
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
        if not done:
            return {'loss': 0.0}
        
        # Monte Carlo return calculation
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize
        
        # Policy gradient update
        states_t = Tensor(np.array(self.episode_states))
        probs = self.policy(states_t)
        
        # Gather log probabilities
        log_probs = np.log(probs.data[np.arange(len(self.episode_actions)), self.episode_actions] + 1e-8)
        log_probs_t = Tensor(log_probs, requires_grad=True, _children=(probs,), _op='log_gather')
        
        def _log_gather_backward():
            parent = list(log_probs_t._prev)[0]
            grad = np.zeros_like(parent.data)
            grad[np.arange(len(self.episode_actions)), self.episode_actions] = log_probs_t.grad
            parent.grad = parent.grad + grad if parent.grad is not None else grad
        
        log_probs_t._backward = _log_gather_backward
        
        loss = -(log_probs_t * Tensor(returns)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return {'loss': loss.item()}
    
    def parameters(self):
        return self.policy.parameters()