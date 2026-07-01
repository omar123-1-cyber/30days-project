"""
Day 5: Tabular Q-Learning
"""

import numpy as np
from .base import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_idx = np.argmax(state)  # one-hot to index
        
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        return int(np.argmax(self.q_table[state_idx]))
    
    def update(self, state, action, reward, next_state, done) -> dict:
        state_idx = np.argmax(state)
        next_state_idx = np.argmax(next_state)
        
        current_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx])
        
        target = reward + (0 if done else self.gamma * next_max_q)
        self.q_table[state_idx, action] += self.lr * (target - current_q)
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {'loss': float((target - current_q) ** 2)}
    
    def parameters(self):
        return []  # Tabular has no neural params