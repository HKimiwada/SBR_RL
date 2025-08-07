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
from sb3_stacking_environment import make_stacking_env
from stable_baselines3 import PPO
import numpy as np

print("🚀 Starting training...")

# Create environment (now returns flattened features)
env = make_stacking_env(
    task_variant='stack_3_bricks',
    use_vision=False,  # Flattened features for MlpPolicy
    reward_scale=10.0
)

print(f"Environment observation space: {env.observation_space}")
print(f"Environment action space: {env.action_space}")

# Create PPO model
model = PPO(
    'MultiInputPolicy', 
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

print("🎯 Training started...")

# Train the model
model.learn(total_timesteps=50_000, progress_bar=True)

# Save model
model.save("ppo_stacking_model")
print("✅ Model saved!")

# Test trained model
print("🔍 Testing trained model...")
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 20 == 0:
        print(f"Step {i}: reward={reward:.3f}")
    
    if terminated or truncated:
        print(f"Episode ended at step {i}")
        break

env.close()
print("✅ Training complete!")