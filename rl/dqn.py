"""
Day 6-7: DQN, Double DQN, Dueling DQN
"""

import numpy as np
from core.tensor import Tensor
from core.nn import Module, Linear, ReLU, Sequential
from core.optim import Adam
from .base import BaseAgent
from .replay_buffer import ReplayBuffer


class QNetwork(Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([Linear(prev_dim, h), ReLU()])
            prev_dim = h
        layers.append(Linear(prev_dim, action_dim))
        self.net = Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DuelingQNetwork(Module):
    """Day 7: Dueling Architecture — value + advantage streams"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.shared = Sequential(
            Linear(state_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU()
        )
        self.value_stream = Sequential(Linear(hidden_dim, 1))
        self.advantage_stream = Sequential(Linear(hidden_dim, action_dim))
    
    def forward(self, x: Tensor) -> Tensor:
        features = self.shared(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        return value + (advantage - advantage.mean(axis=1, keepdims=True))


class DQNAgent(BaseAgent):
    """
    Day 6: Deep Q-Network
    Day 7: Double DQN support added
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, epsilon: float = 1.0, 
                 buffer_size: int = 100000, batch_size: int = 64,
                 target_update_freq: int = 1000, double: bool = False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double = double
        self.update_counter = 0
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self._sync_target()
        
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
    
    def _sync_target(self):
        # Copy weights from q_net to target_net
        for p_q, p_t in zip(self.q_net.parameters(), self.target_net.parameters()):
            p_t.data = p_q.data.copy()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_t = Tensor(state.reshape(1, -1))
        q_values = self.q_net(state_t)
        return int(np.argmax(q_values.data))
    
    def update(self, state, action, reward, next_state, done) -> dict:
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states_t = Tensor(states)
        actions_t = Tensor(actions)
        rewards_t = Tensor(rewards.reshape(-1, 1))
        next_states_t = Tensor(next_states)
        dones_t = Tensor(dones.reshape(-1, 1))
        
        # Current Q values
        current_q = self.q_net(states_t)
        # Gather actions
        current_q = Tensor(
            current_q.data[np.arange(self.batch_size), actions],
            requires_grad=True,
            _children=(current_q,),
            _op='gather'
        )
        
        def _gather_backward():
            if current_q._prev:
                parent = list(current_q._prev)[0]
                grad = np.zeros_like(parent.data)
                grad[np.arange(self.batch_size), actions] = current_q.grad
                parent.grad = parent.grad + grad if parent.grad is not None else grad
        
        current_q._backward = _gather_backward
        
        # Target Q values
        if self.double:
            # Day 7: Double DQN — select action using online net, evaluate using target
            next_q_online = self.q_net(next_states_t).data
            next_actions = np.argmax(next_q_online, axis=1)
            next_q_target = self.target_net(next_states_t).data
            next_q = next_q_target[np.arange(self.batch_size), next_actions]
        else:
            next_q = np.max(self.target_net(next_states_t).data, axis=1)
        
        target = rewards + (1 - dones) * self.gamma * next_q
        target_t = Tensor(target.reshape(-1, 1), requires_grad=False)
        
        # Loss: MSE
        diff = current_q - target_t
        loss = (diff * diff).mean()
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._sync_target()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def parameters(self):
        return self.q_net.parameters()


class DoubleDQNAgent(DQNAgent):
    """Day 7: Double DQN wrapper"""
    def __init__(self, *args, **kwargs):
        kwargs['double'] = True
        super().__init__(*args, **kwargs)


class DuelingDQNAgent(DQNAgent):
    """Day 7: Dueling DQN with DuelingQNetwork"""
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = kwargs.get('batch_size', 64)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        self.double = kwargs.get('double', False)
        self.update_counter = 0
        
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self._sync_target()
        
        self.optimizer = Adam(self.q_net.parameters(), lr=kwargs.get('lr', 3e-4))
        self.buffer = ReplayBuffer(kwargs.get('buffer_size', 100000))