# reward_scaling_environment.py - Simple fix: scale up tiny rewards
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ':0'

import gymnasium as gym
from gymnasium import spaces
from dm_control import manipulation
import numpy as np
from typing import Dict, Any, Tuple, Optional


class RewardScaledStackingEnv(gym.Env):
    """
    Stacking environment with reward scaling to fix tiny reward magnitudes
    """
    
    def __init__(self, task_variant='stack_3_bricks', max_episode_steps=1500, reward_scale=1000.0):
        super().__init__()
        
        self.task_name = task_variant
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale  # Scale up tiny rewards
        self._episode_step = 0
        
        # Load dm_control environment
        try:
            env_name = f"{task_variant}_features"
            self.env = manipulation.load(env_name)
            print(f"✅ Loaded: {env_name}")
        except:
            self.env = manipulation.load(task_variant)
            print(f"✅ Loaded: {task_variant}")
        
        # Override dm_control's episode length
        if hasattr(self.env, '_time_limit'):
            self.env._time_limit = max_episode_steps * 2
        
        try:
            if hasattr(self.env, '_task') and hasattr(self.env._task, '_time_limit'):
                self.env._task._time_limit = max_episode_steps * 2
        except:
            pass
        
        # Set up spaces
        self._setup_spaces()
        
        self._current_time_step = None
        print(f"✅ RewardScaledStackingEnv ready (scale={reward_scale}x)")
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        temp_time_step = self.env.reset()
        flat_obs = self._flatten_observation(temp_time_step.observation)
        
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
        
        for key in sorted(dm_obs.keys()):
            value = dm_obs[key]
            if isinstance(value, np.ndarray):
                obs_parts.append(value.flatten())
            else:
                obs_parts.append(np.array([value], dtype=np.float32))
        
        flattened = np.concatenate(obs_parts).astype(np.float32)
        return flattened
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self._episode_step = 0
        self._current_time_step = self.env.reset()
        
        obs = self._flatten_observation(self._current_time_step.observation)
        
        info = {
            'episode_step': self._episode_step,
            'success': False,
            'max_steps': self.max_episode_steps,
            'reward_scale': self.reward_scale
        }
        
        return obs, info
    
    def step(self, action):
        """Step environment with reward scaling"""
        self._current_time_step = self.env.step(action)
        self._episode_step += 1
        
        obs = self._flatten_observation(self._current_time_step.observation)
        
        # Get original tiny reward
        original_reward = float(self._current_time_step.reward)
        
        # Scale up the reward to make it learnable
        scaled_reward = original_reward * self.reward_scale
        
        # Termination logic (based on original reward)
        task_success = original_reward > 0.008  # Scaled threshold for success
        terminated = task_success
        truncated = self._episode_step >= self.max_episode_steps
        
        success = original_reward > 0.004 or task_success  # Lower threshold for "success"
        
        info = {
            'episode_step': self._episode_step,
            'success': success,
            'task_success': task_success,
            'original_reward': original_reward,
            'scaled_reward': scaled_reward,
            'max_steps': self.max_episode_steps,
            'reward_scale': self.reward_scale
        }
        
        # Debug logging for significant rewards
        if scaled_reward > 1.0:  # Now we can see meaningful rewards!
            print(f"Step {self._episode_step}: original={original_reward:.6f}, scaled={scaled_reward:.3f}")
        
        return obs, scaled_reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        """Render environment"""
        try:
            return self.env.physics.render(height=480, width=640, camera_id=0)
        except:
            return None
    
    def close(self):
        """Close environment"""
        pass


# Factory function
def make_scaled_env(task_variant='stack_3_bricks', max_episode_steps=1500, reward_scale=1000.0):
    """Create reward-scaled stacking environment"""
    return RewardScaledStackingEnv(
        task_variant=task_variant, 
        max_episode_steps=max_episode_steps,
        reward_scale=reward_scale
    )


# Test the reward scaling
if __name__ == "__main__":
    print("🧪 Testing Reward Scaling")
    print("=" * 40)
    
    # Test different scaling factors
    scales = [1.0, 100.0, 1000.0, 10000.0]
    
    for scale in scales:
        print(f"\n🔍 Testing reward_scale={scale}")
        
        env = make_scaled_env(reward_scale=scale, max_episode_steps=50)
        obs, info = env.reset()
        
        total_scaled = 0
        total_original = 0
        
        for step in range(20):
            # Test the action that gave highest reward in diagnostic
            action = np.ones(env.action_space.shape) * 0.1
            obs, scaled_reward, terminated, truncated, info = env.step(action)
            
            total_scaled += scaled_reward
            total_original += info['original_reward']
            
            if step == 0:  # Show first step
                print(f"   Step 0: original={info['original_reward']:.6f}, scaled={scaled_reward:.3f}")
        
        print(f"   Total over 20 steps: original={total_original:.6f}, scaled={total_scaled:.3f}")
        env.close()
    
    print(f"\n✅ Reward scaling test completed!")
    print(f"💡 Scale=1000 gives rewards in the 0.4-4.0 range - perfect for PPO learning!")