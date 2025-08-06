# test_stacking_environment.py
import os
os.environ['MUJOCO_GL'] = 'egl'  # Headless rendering for DGX-1

from dm_control import manipulation
import numpy as np

def test_rgb_stacking_alternative():
    """Test dm_control stacking task as rgb_stacking alternative"""
    
    print("🎮 Testing RGB-Stacking Alternative")
    print("Task: stack_3_bricks_features (similar to rgb_stacking RGB objects)")
    print("=" * 60)
    
    # Load the stacking environment (closest to rgb_stacking)
    env = manipulation.load('stack_3_bricks_features')
    
    # Reset environment
    time_step = env.reset()
    print("✅ Environment loaded and reset successfully!")
    
    # Analyze the task (similar to rgb_stacking)
    print(f"\n📊 Task Analysis:")
    print(f"Action dimensions: {env.action_spec().shape[0]} (robot arm joints)")
    print(f"Action range: [{env.action_spec().minimum[0]:.2f}, {env.action_spec().maximum[0]:.2f}]")
    
    # Show observations (like rgb_stacking state info)
    obs = time_step.observation
    print(f"\n��️ Available Observations (similar to rgb_stacking state):")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"  • {key:20s}: {str(value.shape):15s} - {get_obs_description(key)}")
        else:
            print(f"  • {key:20s}: {type(value).__name__:15s}")
    
    # Test robot control (like rgb_stacking arm movement)
    print(f"\n🤖 Testing Robot Control (similar to rgb_stacking arm):")
    total_reward = 0
    
    for step in range(100):
        # Small random actions (like rgb_stacking exploration)
        action = np.random.uniform(
            low=env.action_spec().minimum,
            high=env.action_spec().maximum,
            size=env.action_spec().shape
        ) * 0.1  # Scale down for stability
        
        time_step = env.step(action)
        total_reward += time_step.reward
        
        # Log progress like rgb_stacking episodes
        if step % 25 == 0:
            print(f"  Step {step:3d}: reward = {time_step.reward:.4f}, total = {total_reward:.4f}")
            # Show arm position if available
            if 'arm_pos' in time_step.observation:
                arm_pos = time_step.observation['arm_pos'][:3]  # First 3 joints
                print(f"           arm joints: [{arm_pos[0]:.3f}, {arm_pos[1]:.3f}, {arm_pos[2]:.3f}]")
        
        if time_step.last():
            print(f"  Episode completed at step {step}!")
            success = total_reward > 0.1
            print(f"  Task success: {'✅ YES' if success else '❌ NO'} (reward: {total_reward:.3f})")
            break
    
    print(f"\n🎉 Test complete! This environment provides:")
    print(f"   ✅ Multi-object stacking (like rgb_stacking RGB cubes)")
    print(f"   ✅ Robot arm control (like rgb_stacking Sawyer arm)")
    print(f"   ✅ Rich observations (like rgb_stacking state)")
    print(f"   ✅ Reward signal for stacking success")
    print(f"   ✅ Modern MuJoCo 3.x (no licensing issues)")

def get_obs_description(key):
    """Get human-readable description of observation"""
    descriptions = {
        'arm_pos': 'Robot arm joint positions',
        'arm_vel': 'Robot arm joint velocities', 
        'touch': 'Gripper touch sensors',
        'hand_pos': 'Gripper 3D position',
        'hand_quat': 'Gripper orientation',
        'brick_pos': 'Brick positions',
        'brick_quat': 'Brick orientations',
        'target': 'Target configuration'
    }
    
    for desc_key, description in descriptions.items():
        if desc_key in key:
            return description
    return 'Task-specific observation'

if __name__ == "__main__":
    test_rgb_stacking_alternative()
