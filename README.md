# NeuroTopology: 30-Day ML + DRL + FEA Challenge

**Day 1: Core Tensor Engine with Autograd**

This repository documents a 30-day journey to build a complete machine learning, deep reinforcement learning, and finite element analysis project from scratch using only NumPy.

## Project Structure
- `core/` — Custom tensor operations and autograd engine
- `rl/` — Deep RL algorithms (DQN, PPO, SAC)
- `fea/` — Finite Element Analysis solver
- `envs/` — RL environments for topology optimization

## Day 1 Features
- Custom `Tensor` class wrapping NumPy arrays
- Reverse-mode automatic differentiation
- Supports: add, sub, mul, div, matmul, pow, relu, sigmoid, tanh, exp, log
- Reduction ops: sum, mean, max
- Shape ops: reshape, transpose
- Broadcasting-aware gradient computation
- Topological sort for backward pass

## How to Run
```bash
pip install -r requirements.txt
python -c "from core.tensor import Tensor; print('Tensor engine ready!')"