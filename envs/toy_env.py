"""
Day 4: Toy Environments for Testing RL Algorithms
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rl.base import BaseEnv


class GridWorldEnv(BaseEnv):
    """Simple 5x5 GridWorld. Agent reaches goal."""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        self._state_dim = size * size
        self._action_dim = 4  # up, down, left, right
    
    def reset(self):
        self.agent_pos = np.array([0, 0])
        return self._get_state()
    
    def _get_state(self):
        state = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        return state
    
    def step(self, action: int):
        # Actions: 0=up, 1=down, 2=left, 3=right
        move = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_pos = self.agent_pos + np.array(move)
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos
        
        reward = -0.1  # step penalty
        done = False
        
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            done = True
        
        return self._get_state(), reward, done, {}
    
    @property
    def state_dim(self): return self._state_dim
    @property
    def action_dim(self): return self._action_dim


class CartPoleLikeEnv(BaseEnv):
    """
    Day 4: Simplified CartPole for continuous control testing.
    State: [position, velocity, angle, angular_velocity]
    Action: 0 (push left), 1 (push right)
    """
    
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        self._state_dim = 4
        self._action_dim = 2
    
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.astype(np.float32)
    
    def step(self, action: int):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        self.state = self.state + self.tau * np.array([x_dot, xacc, theta_dot, thetaacc])
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )
        
        reward = 1.0
        return self.state.astype(np.float32), reward, done, {}
    
    @property
    def state_dim(self): return self._state_dim
    @property
    def action_dim(self): return self._action_dim