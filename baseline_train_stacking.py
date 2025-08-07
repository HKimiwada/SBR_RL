"""
Criteria for fair baseline:
1. Exact same environment (reward function, physics, action space, episode length, etc...)
2. Same evaluation protocol (same number of episodes, same seed)
3. Same observation space (unify use_vision)
4. Equal Environment Steps
5. Similar network architecture (same number of layers, same number of units per layer)

Baseline Models:
1. Random Policy (Implemented in test_train_stacking.py)
2. Soft Actor-Critic (SAC: implemented using stable_baselines3)
3. Proximal Policy Optimization (PPO: implemented using stable_baselines3)
"""
import wandb
from stable_baselines3 import SAC, PPO