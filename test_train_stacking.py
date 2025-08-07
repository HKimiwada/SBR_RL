# train_stacking.py
# Use conda environment modern_stacking 
# use_vision = False: State Mode (Direct Access to all object positions: Faster to train, unrealistic)
# use_vision = True: Vision Mode (RGB images, slower to train, need CNN, but more realistic)
"""
Output:
(modern_stacking) hiroki_kimiwada@dgx-1:~/SBR_RL$ python test_train_stacking.py
Simple Training Loop
========================================
✅ Loaded: stack_3_bricks_features
✅ GPU rendering working properly
🎮 RGBStackingAlternative Environment (GPU Rendering)
   Task: stack_3_bricks
   Vision: Disabled
   Render size: (480, 640)

📊 Episode 1/10
   Step 0: reward = 0.0000
   Step 50: reward = 0.0000
   Step 100: reward = 0.0000
   Step 150: reward = 0.0000
   Total reward: 0.0029
   Success: ❌

📊 Episode 2/10
   Step 0: reward = 0.0000
   Step 50: reward = 0.0000
   Step 100: reward = 0.0000
   Step 150: reward = 0.0000
   Total reward: 0.0016
   Success: ❌

📊 Episode 3/10
   Step 0: reward = 0.0005
   Step 50: reward = 0.0005
   Step 100: reward = 0.0005
   Step 150: reward = 0.0005
   Total reward: 0.0980
   Success: ❌

📊 Episode 4/10
   Step 0: reward = 0.0000
   Step 50: reward = 0.0000
   Step 100: reward = 0.0000
   Step 150: reward = 0.0000
   Total reward: 0.0069
   Success: ❌

📊 Episode 5/10
   Step 0: reward = 0.0000
   Step 50: reward = 0.0000
   Step 100: reward = 0.0000
   Step 150: reward = 0.0000
   Total reward: 0.0000
   Success: ❌

📊 Episode 6/10
   Step 0: reward = 0.0000
   Step 50: reward = 0.0000
   Step 100: reward = 0.0000
   Step 150: reward = 0.0000
   Total reward: 0.0000
   Success: ❌

📊 Episode 7/10
   Step 0: reward = 0.0001
   Step 50: reward = 0.0001
   Step 100: reward = 0.0001
   Step 150: reward = 0.0001
   Total reward: 0.0102
   Success: ❌

📊 Episode 8/10
   Step 0: reward = 0.0009
   Step 50: reward = 0.0009
   Step 100: reward = 0.0009
   Step 150: reward = 0.0009
   Total reward: 0.1715
   Success: ❌

📊 Episode 9/10
   Step 0: reward = 0.0001
   Step 50: reward = 0.0001
   Step 100: reward = 0.0001
   Step 150: reward = 0.0001
   Total reward: 0.0130
   Success: ❌

📊 Episode 10/10
   Step 0: reward = 0.0000
   Step 50: reward = 0.0000
   Step 100: reward = 0.0000
   Step 150: reward = 0.0000
   Total reward: 0.0028
   Success: ❌

📈 Training Summary:
   Average reward: 0.0307
   Best episode: 0.1715
   Success rate: 1/10
"""

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