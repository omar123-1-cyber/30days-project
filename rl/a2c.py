"""
Day 9: A2C — Advantage Actor-Critic (synchronous)
"""

import numpy as np
from core.tensor import Tensor
from core.nn import Module, Linear, ReLU, Sequential
from core.optim import Adam
from .base import BaseAgent


class ActorCriticNetwork(Module):
    """Shared backbone with actor and critic heads."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.shared = Sequential(
            Linear(state_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU()
        )
        self.actor = Linear(hidden_dim, action_dim)
        self.critic = Linear(hidden_dim, 1)
    
    def forward(self, x: Tensor) -> tuple:
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        
        # Softmax for policy
        shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        policy = Tensor(probs, requires_grad=True, _children=(logits,), _op='softmax')
        
        return policy, value


class A2CAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_t = Tensor(state.reshape(1, -1))
        policy, _ = self.network(state_t)
        probs = policy.data[0]
        
        if training:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = np.argmax(probs)
        return int(action)
    
    def update(self, state, action, reward, next_state, done) -> dict:
        state_t = Tensor(state.reshape(1, -1))
        next_state_t = Tensor(next_state.reshape(1, -1))
        
        policy, value = self.network(state_t)
        _, next_value = self.network(next_state_t)
        
        # TD target
        target = reward + (0 if done else self.gamma * next_value.data[0, 0])
        advantage = target - value.data[0, 0]
        
        # Critic loss (MSE)
        value_loss = (value - Tensor([[target]])) ** 2
        
        # Actor loss (policy gradient with advantage)
        log_prob = np.log(policy.data[0, action] + 1e-8)
        log_prob_t = Tensor(np.array([log_prob]), requires_grad=True, _children=(policy,), _op='log_gather')
        
        def _backward():
            parent = list(log_prob_t._prev)[0]
            grad = np.zeros_like(parent.data)
            grad[0, action] = log_prob_t.grad[0]
            parent.grad = parent.grad + grad if parent.grad is not None else grad
        
        log_prob_t._backward = _backward
        
        policy_loss = -log_prob_t * Tensor([advantage])
        
        # Entropy bonus
        entropy = -np.sum(policy.data[0] * np.log(policy.data[0] + 1e-8))
        entropy_t = Tensor([entropy], requires_grad=False)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_t
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'value_loss': float(value_loss.item()), 'advantage': float(advantage)}
    
    def parameters(self):
        return self.network.parameters()