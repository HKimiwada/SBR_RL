# test_sb3_integration.py - Test StableBaselines3 integration
from sb3_stacking_environment import make_simple_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import torch

def test_environment_compatibility():
    """Test basic environment compatibility"""
    print("🧪 Testing Environment Compatibility")
    print("=" * 50)
    
    # Create environment
    env = make_simple_env('stack_3_bricks')
    
    # Test 1: Basic functionality
    print("📋 Test 1: Basic Environment Functions")
    obs, info = env.reset()
    print(f"   ✅ Reset: obs shape = {obs.shape}, type = {type(obs)}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✅ Step: obs shape = {obs.shape}, reward = {reward:.3f}")
    
    # Test 2: Observation space compatibility
    print("\n📊 Test 2: Observation Space Check")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) == 1:
        print("   ✅ Observation space is properly flattened for MlpPolicy")
    else:
        print("   ❌ Observation space issue!")
        return False
    
    # Test 3: SB3 environment checker
    print("\n🔍 Test 3: StableBaselines3 Environment Checker")
    try:
        check_env(env, warn=True)
        print("   ✅ Environment passes SB3 compatibility check")
    except Exception as e:
        print(f"   ❌ Environment checker failed: {e}")
        return False
    
    env.close()
    return True

def test_ppo_creation():
    """Test PPO model creation"""
    print("\n🤖 Testing PPO Model Creation")
    print("=" * 40)
    
    env = make_simple_env('stack_3_bricks')
    
    try:
        # Test device selection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        # Create PPO model
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=256,  # Small for testing
            batch_size=32,
            verbose=1,
            device=device
        )
        print("   ✅ PPO model created successfully")
        
        # Test model prediction
        obs, info = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        print(f"   ✅ Model prediction: action shape = {action.shape}")
        
        env.close()
        return model, env
        
    except Exception as e:
        print(f"   ❌ PPO creation failed: {e}")
        env.close()
        return None, None

def test_short_training():
    """Test a very short training session"""
    print("\n🎯 Testing Short Training Session")
    print("=" * 40)
    
    env = make_simple_env('stack_3_bricks')
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            verbose=0,  # Quiet for test
            device=device
        )
        
        print("   🏃 Running 1000 training steps...")
        model.learn(total_timesteps=1000, progress_bar=False)
        print("   ✅ Short training completed successfully")
        
        # Test trained model
        print("   🔍 Testing trained model...")
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(20):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"   Episode ended at step {step}")
                break
        
        print(f"   ✅ Total reward over {step+1} steps: {total_reward:.3f}")
        
        # Save and load test
        model.save("test_model")
        print("   ✅ Model saved successfully")
        
        loaded_model = PPO.load("test_model")
        print("   ✅ Model loaded successfully")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False

def test_multiple_episodes():
    """Test running multiple episodes"""
    print("\n🔄 Testing Multiple Episodes")
    print("=" * 35)
    
    env = make_simple_env('stack_3_bricks', max_episode_steps=50)  # Short episodes for testing
    
    try:
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(3):
            print(f"   Episode {episode + 1}/3:")
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = env.action_space.sample() * 0.1  # Small random actions
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success = info.get('success', False)
            
            print(f"      Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {success}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"   ✅ Average reward: {avg_reward:.3f}")
        print(f"   ✅ Average length: {avg_length:.1f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Multiple episodes test failed: {e}")
        env.close()
        return False

def main():
    """Run all tests"""
    print("🚀 StableBaselines3 Integration Test")
    print("=" * 50)
    
    # Test 1: Environment compatibility
    if not test_environment_compatibility():
        print("\n❌ Environment compatibility test failed!")
        return False
    
    # Test 2: PPO model creation
    model, env = test_ppo_creation()
    if model is None:
        print("\n❌ PPO creation test failed!")
        return False
    
    # Test 3: Short training
    if not test_short_training():
        print("\n❌ Training test failed!")
        return False
    
    # Test 4: Multiple episodes
    if not test_multiple_episodes():
        print("\n❌ Multiple episodes test failed!")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Environment is fully compatible with StableBaselines3")
    print("✅ PPO model creation works")
    print("✅ Training works")
    print("✅ Model save/load works")
    print("✅ Multiple episodes work")
    print("\n💡 You can now run full training with confidence!")
    print("   Recommended next step: Create a full training script")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎯 Ready for full training! Example:")
        print("   from simple_stacking_env import make_simple_env")
        print("   from stable_baselines3 import PPO")
        print("   env = make_simple_env()")
        print("   model = PPO('MlpPolicy', env, verbose=1)")
        print("   model.learn(total_timesteps=100_000)")
    else:
        print("\n💥 Some tests failed. Check the errors above.")