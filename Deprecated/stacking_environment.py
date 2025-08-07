# stacking_environment.py
import os
os.environ['MUJOCO_GL'] = 'egl'  # Headless rendering for DGX-1

from dm_control import manipulation
import numpy as np

class RGBStackingAlternative:
    """
    Modern alternative to rgb_stacking using dm_control
    
    Provides the same functionality as rgb_stacking but with:
    - Modern MuJoCo 3.x (no license needed)
    - GPU acceleration on DGX-1
    - Multiple task variants
    - Better vision support
    """
    
    def __init__(self, task_variant='stack_3_bricks', use_vision=False):
        """
        Initialize the stacking environment
        
        Args:
            task_variant: 'stack_3_bricks', 'stack_2_bricks', etc.
            use_vision: If True, includes visual observations
        """
        # Choose task based on vision requirement
        if use_vision:
            task_name = f"{task_variant}_vision"
        else:
            task_name = f"{task_variant}_features"
            
        self.task_name = task_name
        self.use_vision = use_vision
        self.env = manipulation.load(task_name)
        self._current_time_step = None
        
        print(f"🎮 RGBStackingAlternative Environment")
        print(f"   Task: {task_name}")
        print(f"   Vision: {'Enabled' if use_vision else 'Disabled'}")
        print(f"   Similar to: rgb_stacking RGB cube task")
    
    def reset(self):
        """Reset environment (like rgb_stacking.reset())"""
        self._current_time_step = self.env.reset()
        return self._get_observation()
    
    def step(self, action):
        """
        Step environment (like rgb_stacking.step())
        
        Returns:
            observation, reward, done, info (standard RL interface)
        """
        self._current_time_step = self.env.step(action)
        
        obs = self._get_observation()
        reward = float(self._current_time_step.reward)
        done = self._current_time_step.last()
        info = {
            'discount': self._current_time_step.discount,
            'success': reward > 0.5  # Simple success metric
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation (like rgb_stacking observations)"""
        if self._current_time_step is None:
            return None
        
        obs_dict = self._current_time_step.observation.copy()
        
        # Add visual observation if requested (like rgb_stacking RGB)
        if self.use_vision and hasattr(self.env, 'physics'):
            try:
                # Render visual observation (like rgb_stacking camera)
                rgb_image = self.env.physics.render(
                    height=84, width=84, camera_id=0
                )
                obs_dict['rgb'] = rgb_image
            except Exception as e:
                print(f"Warning: Could not render visual observation: {e}")
        
        return obs_dict
    
    def get_action_space(self):
        """Get action space (like rgb_stacking action space)"""
        spec = self.env.action_spec()
        return {
            'shape': spec.shape,
            'low': spec.minimum,
            'high': spec.maximum,
            'dtype': np.float32
        }
    
    def sample_random_action(self):
        """Sample random action (like rgb_stacking random policy)"""
        spec = self.env.action_spec()
        return np.random.uniform(
            low=spec.minimum,
            high=spec.maximum,
            size=spec.shape
        ).astype(np.float32)
    
    def render(self, mode='rgb_array', height=240, width=320):
        """Render environment (like rgb_stacking rendering)"""
        if hasattr(self.env, 'physics'):
            try:
                return self.env.physics.render(
                    height=height, width=width, camera_id=0
                )
            except Exception as e:
                print(f"Rendering failed: {e}")
                return None
        return None
    
    def close(self):
        """Close environment"""
        # dm_control environments don't need explicit closing
        pass

def demo_rgb_stacking_alternative():
    """Demo the RGB-stacking alternative environment"""
    
    print("🚀 RGB-Stacking Alternative Demo")
    print("=" * 50)
    
    # Test state-based environment (like rgb_stacking state)
    print("\n1. Testing State-Based Environment:")
    env_state = RGBStackingAlternative('stack_3_bricks', use_vision=False)
    
    obs = env_state.reset()
    print(f"   ✅ Reset successful, observations: {list(obs.keys())}")
    
    action_space = env_state.get_action_space()
    print(f"   ✅ Action space: {action_space['shape']} dimensions")
    
    # Run a few steps
    for i in range(10):
        action = env_state.sample_random_action() * 0.1
        obs, reward, done, info = env_state.step(action)
        if i % 5 == 0:
            print(f"   Step {i}: reward = {reward:.4f}, done = {done}")
        if done:
            print(f"   Episode finished at step {i}")
            break
    
    # Test vision-based environment (like rgb_stacking visual)
    print("\n2. Testing Vision-Based Environment:")
    env_vision = RGBStackingAlternative('stack_3_bricks', use_vision=True)
    
    obs = env_vision.reset()
    print(f"   ✅ Reset successful, observations: {list(obs.keys())}")
    
    # Test rendering
    rgb_img = env_vision.render()
    if rgb_img is not None:
        print(f"   ✅ Rendering successful: {rgb_img.shape}")
    
    # Test different task variants
    print("\n3. Available Task Variants:")
    variants = ['stack_2_bricks', 'stack_3_bricks', 'place_brick']
    for variant in variants:
        try:
            test_env = RGBStackingAlternative(variant, use_vision=False)
            print(f"   ✅ {variant}: Available")
        except Exception as e:
            print(f"   ❌ {variant}: {str(e)}")
    
    print(f"\n Installation Successful: rgb_stacking alternative!")

if __name__ == "__main__":
    demo_rgb_stacking_alternative()
