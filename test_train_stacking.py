# train_stacking.py
# Use conda environment modern_stacking 
# use_vision = False: State Mode (Direct Access to all object positions: Faster to train, unrealistic)
# use_vision = True: Vision Mode (RGB images, slower to train, need CNN, but more realistic)
import os
os.environ['MUJOCO_GL'] = 'egl'

from stacking_environment import RGBStackingAlternative
import numpy as np

def simple_training_loop():
    """Simple training loop"""
    
    print("Simple Training Loop")
    print("=" * 40)
    
    # Create environment
    env = RGBStackingAlternative('stack_3_bricks', use_vision=False)
    
    # Training parameters
    num_episodes = 10
    max_steps = 200
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Simple random policy
            action = env.sample_random_action() * 0.2  # Small actions
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if step % 50 == 0:
                print(f"   Step {step}: reward = {reward:.4f}")
            
            if done:
                print(f"   Episode finished at step {step}")
                break
        
        episode_rewards.append(total_reward)
        success = info.get('success', False) if 'info' in locals() else total_reward > 0.1
        
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Success: {'✅' if success else '❌'}")
    
    print(f"\n📈 Training Summary:")
    print(f"   Average reward: {np.mean(episode_rewards):.4f}")
    print(f"   Best episode: {max(episode_rewards):.4f}")
    print(f"   Success rate: {sum(r > 0.1 for r in episode_rewards)}/{num_episodes}")
    
    print(f"\n🎉 Training complete! Ready for advanced RL algorithms!")

if __name__ == "__main__":
    simple_training_loop()