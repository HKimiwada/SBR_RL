# reward_diagnostic.py - Analyze the reward structure and task behavior
import numpy as np
from sb3_stacking_environment import make_simple_env
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_reward_structure():
    """Analyze what gives rewards in the stacking task"""
    print("🔍 Analyzing Reward Structure")
    print("=" * 50)
    
    env = make_simple_env('stack_3_bricks', max_episode_steps=500)  # Shorter for analysis
    
    # Test different action strategies
    strategies = {
        'random_large': lambda: env.action_space.sample(),
        'random_small': lambda: env.action_space.sample() * 0.1,
        'zero_action': lambda: np.zeros(env.action_space.shape),
        'constant_small': lambda: np.ones(env.action_space.shape) * 0.05,
    }
    
    results = {}
    
    for strategy_name, action_fn in strategies.items():
        print(f"\n📊 Testing strategy: {strategy_name}")
        
        episode_rewards = []
        step_rewards = []
        episode_lengths = []
        
        for episode in range(5):  # Test 5 episodes per strategy
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(500):
                action = action_fn()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                step_rewards.append(reward)
                
                # Print any non-zero rewards immediately
                if reward != 0:
                    print(f"   🎯 Non-zero reward at step {step}: {reward:.6f}")
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            print(f"   Episode {episode+1}: total_reward={episode_reward:.6f}, length={episode_steps}")
        
        results[strategy_name] = {
            'episode_rewards': episode_rewards,
            'step_rewards': step_rewards,
            'episode_lengths': episode_lengths,
            'max_reward': max(step_rewards) if step_rewards else 0,
            'non_zero_count': sum(1 for r in step_rewards if r != 0)
        }
        
        print(f"   Summary: max_reward={results[strategy_name]['max_reward']:.6f}, "
              f"non_zero_steps={results[strategy_name]['non_zero_count']}")
    
    env.close()
    return results

def analyze_observation_space():
    """Analyze what information is available in observations"""
    print("\n🔍 Analyzing Observation Space")
    print("=" * 40)
    
    env = make_simple_env('stack_3_bricks')
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Observation mean: {obs.mean():.3f}")
    print(f"Observation std: {obs.std():.3f}")
    
    # Check if there are obvious patterns
    print(f"\nFirst 20 observation values:")
    print(obs[:20])
    
    # Take a few steps and see how observations change
    print(f"\nObservation changes over 3 steps:")
    for step in range(3):
        action = env.action_space.sample() * 0.1
        new_obs, reward, terminated, truncated, info = env.step(action)
        change = np.abs(new_obs - obs).mean()
        print(f"Step {step+1}: avg_change={change:.6f}, reward={reward:.6f}")
        obs = new_obs
    
    env.close()

def test_action_effects():
    """Test how different actions affect the environment"""
    print("\n🔍 Testing Action Effects")
    print("=" * 35)
    
    env = make_simple_env('stack_3_bricks')
    
    # Test action space
    print(f"Action space: {env.action_space}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    
    # Test specific actions
    test_actions = [
        ("zero", np.zeros(env.action_space.shape)),
        ("small_positive", np.ones(env.action_space.shape) * 0.1),
        ("small_negative", np.ones(env.action_space.shape) * -0.1),
        ("random_sample", env.action_space.sample()),
    ]
    
    for action_name, action in test_actions:
        print(f"\nTesting {action_name} action: {action}")
        
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if reward != 0:
                print(f"  Step {step}: reward={reward:.6f}")
        
        print(f"  Total reward over 10 steps: {total_reward:.6f}")
    
    env.close()

def check_dm_control_task_details():
    """Check what we can learn about the underlying dm_control task"""
    print("\n🔍 Checking DM Control Task Details")
    print("=" * 45)
    
    env = make_simple_env('stack_3_bricks')
    
    # Try to access underlying dm_control environment
    if hasattr(env, 'env'):
        dm_env = env.env
        print(f"DM Control environment type: {type(dm_env)}")
        
        # Check if we can get task information
        if hasattr(dm_env, '_task'):
            task = dm_env._task
            print(f"Task type: {type(task)}")
            
            # Try to get reward function details
            if hasattr(task, 'get_reward'):
                obs, info = env.reset()
                # Take a few steps and check reward calculation
                for step in range(5):
                    action = env.action_space.sample() * 0.1
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step {step}: reward={reward:.6f}")
        
        # Check physics simulation
        if hasattr(dm_env, 'physics'):
            physics = dm_env.physics
            print(f"Physics type: {type(physics)}")
            
            # Try to get object positions if possible
            try:
                # This might work depending on the task
                if hasattr(physics, 'data'):
                    print(f"Physics data available")
                    # Could analyze object positions here
            except:
                pass
    
    env.close()

def suggest_reward_shaping():
    """Suggest potential reward shaping strategies"""
    print("\n💡 Reward Shaping Suggestions")
    print("=" * 40)
    
    suggestions = [
        "1. Distance-based rewards: Reward getting gripper closer to blocks",
        "2. Contact rewards: Small reward for touching blocks",
        "3. Pickup rewards: Medium reward for lifting blocks",
        "4. Height rewards: Reward for moving blocks upward",
        "5. Proximity rewards: Reward for bringing blocks closer together",
        "6. Stacking progress: Incremental rewards for partial stacking",
        "7. Exploration bonus: Encourage diverse actions early in training"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

def main():
    """Run complete diagnostic analysis"""
    print("🚀 Stacking Environment Reward Diagnostic")
    print("=" * 60)
    
    try:
        # Run all diagnostic tests
        results = analyze_reward_structure()
        analyze_observation_space()
        test_action_effects()
        check_dm_control_task_details()
        suggest_reward_shaping()
        
        # Summary
        print("\n📋 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        
        total_non_zero = sum(r['non_zero_count'] for r in results.values())
        total_steps = sum(len(r['step_rewards']) for r in results.values())
        
        print(f"Total steps tested: {total_steps}")
        print(f"Steps with non-zero reward: {total_non_zero}")
        print(f"Reward sparsity: {(1 - total_non_zero/total_steps)*100:.1f}% of steps have zero reward")
        
        if total_non_zero == 0:
            print("\n❌ CRITICAL: No non-zero rewards found!")
            print("This confirms the sparse reward problem.")
            print("We need reward shaping or curriculum learning.")
        else:
            print(f"\n✅ Found {total_non_zero} steps with rewards")
            max_reward = max(r['max_reward'] for r in results.values())
            print(f"Maximum reward achieved: {max_reward:.6f}")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()