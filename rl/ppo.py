"""
Day 10-11: PPO — Proximal Policy Optimization with GAE
"""

import numpy as np
from core.tensor import Tensor
from core.nn import Module, Linear, ReLU, Sequential
from core.optim import Adam
from .base import BaseAgent


class PPONetwork(Module):
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
        
        shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        policy = Tensor(probs, requires_grad=True, _children=(logits,), _op='softmax')
        
        return policy, value


class PPOAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, epochs_per_update: int = 4,
                 batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = Adam(self.network.parameters(), lr=lr)
        
        # Trajectory buffer
        self.trajectory = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_t = Tensor(state.reshape(1, -1))
        policy, _ = self.network(state_t)
        probs = policy.data[0]
        
        if training:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = np.argmax(probs)
        return int(action)
    
    def store_transition(self, state, action, reward, next_state, done, log_prob=None):
        self.trajectory.append((state, action, reward, next_state, done))
    
    def update(self, state=None, action=None, reward=None, next_state=None, done=None) -> dict:
        # If single transition, store it
        if state is not None:
            self.store_transition(state, action, reward, next_state, done)
            if not done and len(self.trajectory) < self.batch_size:
                return {'loss': 0.0}
        
        if len(self.trajectory) == 0:
            return {'loss': 0.0}
        
        # Unpack trajectory
        states = np.array([t[0] for t in self.trajectory], dtype=np.float32)
        actions = np.array([t[1] for t in self.trajectory], dtype=np.int64)
        rewards = np.array([t[2] for t in self.trajectory], dtype=np.float32)
        next_states = np.array([t[3] for t in self.trajectory], dtype=np.float32)
        dones = np.array([t[4] for t in self.trajectory], dtype=np.float32)
        
        # Compute values and advantages using GAE
        states_t = Tensor(states)
        next_states_t = Tensor(next_states)
        
        _, values = self.network(states_t)
        _, next_values = self.network(next_states_t)
        
        values_np = values.data.flatten()
        next_values_np = next_values.data.flatten()
        
        # GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_values_np[t]
            else:
                next_val = values_np[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values_np[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values_np
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old policy probabilities
        policy_old, _ = self.network(states_t)
        old_probs = policy_old.data[np.arange(len(actions)), actions]
        
        # PPO update epochs
        total_loss = 0
        for _ in range(self.epochs_per_update):
            policy, value = self.network(states_t)
            
            # New probabilities
            new_probs = policy.data[np.arange(len(actions)), actions]
            ratio = new_probs / (old_probs + 1e-8)
            
            # Clipped surrogate objective
            adv_t = Tensor(advantages)
            ratio_t = Tensor(ratio, requires_grad=True, _children=(policy,), _op='gather_ratio')
            
            def _ratio_backward():
                parent = list(ratio_t._prev)[0]
                grad = np.zeros_like(parent.data)
                grad[np.arange(len(actions)), actions] = ratio_t.grad
                parent.grad = parent.grad + grad if parent.grad is not None else grad
            
            ratio_t._backward = _ratio_backward
            
            surr1 = ratio_t * adv_t
            surr2 = Tensor(np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)) * adv_t
            policy_loss = -Tensor(np.minimum(surr1.data, surr2.data))
            