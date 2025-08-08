# sb3_stacking_environment.py (Environment that inherits from gym.Env for StableBaselines3)
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ':0'

import gymnasium as gym
from gymnasium import spaces
from dm_control import manipulation
import numpy as np
from typing import Dict, Any, Tuple, Optional


class SimpleStackingEnv(gym.Env):
    """
    Simple, clean StableBaselines3 compatible stacking environment
    """
    
    def __init__(self, task_variant='stack_3_bricks', max_episode_steps=1500):
        super().__init__()
        
        self.task_name = task_variant
        self.max_episode_steps = max_episode_steps
        self._episode_step = 0
        
        # Load dm_control environment
        try:
            env_name = f"{task_variant}_features"
            self.env = manipulation.load(env_name)
            print(f"✅ Loaded: {env_name}")
        except:
            self.env = manipulation.load(task_variant)
            print(f"✅ Loaded: {task_variant}")
        
        # Set up spaces
        self._setup_spaces()
        
        self._current_time_step = None
        print(f"✅ SimpleStackingEnv ready with flattened observations")
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        # Get sample observation to calculate flattened size
        temp_time_step = self.env.reset()
        flat_obs = self._flatten_observation(temp_time_step.observation)
        
        # Create flattened observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )
        
        print(f"🔧 Action space: {self.action_space.shape}")
        print(f"🔧 Observation space: {self.observation_space.shape}")
    
    def _flatten_observation(self, dm_obs: Dict) -> np.ndarray:
        """Flatten dm_control observation dict into single array"""
        obs_parts = []
        
        # Sort keys for consistent ordering
        for key in sorted(dm_obs.keys()):
            value = dm_obs[key]
            if isinstance(value, np.ndarray):
                obs_parts.append(value.flatten())
            else:
                obs_parts.append(np.array([value], dtype=np.float32))
        
        # Concatenate all parts
        flattened = np.concatenate(obs_parts).astype(np.float32)
        return flattened
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self._episode_step = 0
        self._current_time_step = self.env.reset()
        
        # Get flattened observation
        obs = self._flatten_observation(self._current_time_step.observation)
        
        info = {
            'episode_step': self._episode_step,
            'success': False
        }
        
        return obs, info
    
    def step(self, action):
        """Step environment"""
        self._current_time_step = self.env.step(action)
        self._episode_step += 1
        
        # Get flattened observation
        obs = self._flatten_observation(self._current_time_step.observation)
        
        reward = float(self._current_time_step.reward)
        terminated = self._current_time_step.last()
        truncated = self._episode_step >= self.max_episode_steps
        
        info = {
            'episode_step': self._episode_step,
            'success': reward > 0.5,
            'discount': float(self._current_time_step.discount)
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        """Render environment"""
        try:
            return self.env.physics.render(height=480, width=640, camera_id=0)
        except:
            return None
    
    def close(self):
        """Close environment"""
        pass


# Simple factory function
def make_simple_env(task_variant='stack_3_bricks', max_episode_steps=1500):
    """Create simple stacking environment"""
    return SimpleStackingEnv(task_variant=task_variant, max_episode_steps=max_episode_steps)


# Test the simple environment
if __name__ == "__main__":
    print("🧪 Testing Simple Environment")
    print("=" * 50)
    
    # Create environment
    env = make_simple_env()
    
    # Test reset
    obs, info = env.reset()
    print(f"Reset: obs shape = {obs.shape}, info = {info}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step: obs shape = {obs.shape}, reward = {reward}")
    
    # Test action and observation spaces
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    env.close()
    print("✅ Simple environment test passed!")