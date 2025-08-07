from sb3_stacking_environment import make_simple_env
from stable_baselines3 import PPO

# Create environment
env = make_simple_env('stack_3_bricks', max_episode_steps=250)

# Create model (use CPU for MLP as recommended)
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1,
    device='cpu',  # CPU is typically faster for MlpPolicy
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

# Train for 50k steps (about 10-15 minutes)
print("🎯 Starting training...")
model.learn(total_timesteps=50_000, progress_bar=True)

# Save model
model.save("ppo_stacking_50k")
print("✅ Model saved!")

env.close()