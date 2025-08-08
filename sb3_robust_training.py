# robust_training.py - Simple, reliable training without complex callbacks
# robust_training_fixed.py - Fixed version with proper episode buffer handling
import os
import torch
import wandb
import numpy as np
import time
from pathlib import Path
from sb3_stacking_environment import make_simple_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id=0):
    """Create a single environment"""
    def _init():
        set_random_seed(env_id)
        env = make_simple_env('stack_3_bricks', max_episode_steps=1500)
        env = Monitor(env)
        return env
    return _init

def create_env(n_envs=8):
    """Create vectorized environment"""
    print(f"🏭 Creating {n_envs} environments...")
    env_fns = [make_env(env_id=i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    print(f"✅ Created {n_envs} environments")
    return env

def safe_get_episode_stats(ep_info_buffer, max_episodes=50):
    """Safely extract episode statistics from buffer"""
    if not ep_info_buffer or len(ep_info_buffer) == 0:
        return [], []
    
    # Convert to list and get recent episodes
    episode_list = list(ep_info_buffer)
    recent_episodes = episode_list[-max_episodes:] if len(episode_list) > max_episodes else episode_list
    
    # Extract rewards and lengths safely
    rewards = []
    lengths = []
    
    for ep in recent_episodes:
        if isinstance(ep, dict):
            if 'r' in ep:
                rewards.append(ep['r'])
            if 'l' in ep:
                lengths.append(ep['l'])
    
    return rewards, lengths

def main():
    """Main training function - simple and robust"""
    print("🚀 Robust Training Script (FIXED)")
    print("=" * 50)
    
    # ============ CONFIGURATION ============
    n_envs = 16                   # Number of parallel environments
    total_timesteps = 500_000     # Total training steps
    device = 'cpu'                # Use CPU for MlpPolicy
    
    # Directories
    output_dir = Path("./training_results")
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    
    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    print(f"📊 Configuration:")
    print(f"   Environments: {n_envs}")
    print(f"   Total steps: {total_timesteps:,}")
    print(f"   Expected time: ~{total_timesteps // (n_envs * 1000)} minutes")
    
    # ============ SETUP WANDB ============
    run_name = f"robust_training_fixed_{int(time.time())}"
    
    wandb.init(
        project="stacking-robot-sb3",
        name=run_name,
        config={
            "algorithm": "PPO",
            "n_envs": n_envs,
            "total_timesteps": total_timesteps,
            "device": device
        }
    )
    print(f"🔍 WandB: {wandb.run.url}")
    
    # ============ CREATE ENVIRONMENTS ============
    train_env = create_env(n_envs)
    eval_env = create_env(1)
    
    # ============ CREATE MODEL ============
    print(f"🤖 Creating PPO model...")
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
        verbose=1,
        tensorboard_log=str(logs_dir),
    )
    
    print("✅ Model created")
    
    # ============ SIMPLE CALLBACKS ============
    # Only use essential callbacks to avoid errors
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=25_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=str(models_dir),
        name_prefix="checkpoint",
        verbose=1
    )
    
    # ============ TRAINING WITH FIXED LOGGING ============
    print(f"🎯 Starting training...")
    start_time = time.time()
    
    # Training loop with manual progress tracking
    steps_per_update = n_envs * 2048  # n_envs * n_steps
    total_updates = total_timesteps // steps_per_update
    
    try:
        for update in range(total_updates):
            # Train for one update cycle
            model.learn(
                total_timesteps=steps_per_update,
                callback=[eval_callback, checkpoint_callback] if update % 5 == 0 else None,
                reset_num_timesteps=False,  # Don't reset timestep counter
                progress_bar=False  # We'll show our own progress
            )
            
            # FIXED: Safe logging every few updates
            if update % 5 == 0:  # Log every 5 updates
                current_timesteps = (update + 1) * steps_per_update
                progress = (update + 1) / total_updates
                
                # Safely get recent episode stats
                rewards, lengths = safe_get_episode_stats(model.ep_info_buffer, max_episodes=50)
                
                if rewards and lengths:
                    mean_reward = np.mean(rewards)
                    mean_length = np.mean(lengths)
                    
                    # Log to WandB
                    wandb.log({
                        "train/timesteps": current_timesteps,
                        "train/progress": progress,
                        "train/mean_reward": mean_reward,
                        "train/mean_length": mean_length,
                        "train/episodes": len(rewards)
                    })
                    
                    print(f"Update {update+1}/{total_updates} "
                          f"({progress:.1%}) - "
                          f"Steps: {current_timesteps:,} - "
                          f"Reward: {mean_reward:.3f} - "
                          f"Episodes: {len(rewards)}")
                else:
                    print(f"Update {update+1}/{total_updates} ({progress:.1%}) - Steps: {current_timesteps:,} - No episodes yet")
        
        # Final model save
        final_model_path = models_dir / "final_model"
        model.save(str(final_model_path))
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"   Duration: {training_time/60:.1f} minutes")
        print(f"   Final model: {final_model_path}.zip")
        
        # Final stats
        rewards, lengths = safe_get_episode_stats(model.ep_info_buffer, max_episodes=100)
        if rewards:
            final_mean = np.mean(rewards)
            wandb.log({
                "final/mean_reward": final_mean,
                "final/training_time_minutes": training_time / 60,
                "final/total_episodes": len(rewards)
            })
            print(f"   Final mean reward: {final_mean:.3f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted!")
        interrupted_path = models_dir / "interrupted_model"
        model.save(str(interrupted_path))
        print(f"   Model saved: {interrupted_path}.zip")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        error_path = models_dir / "error_model"
        model.save(str(error_path))
        print(f"   Model saved: {error_path}.zip")
        
    finally:
        # Clean shutdown
        try:
            train_env.close()
            eval_env.close()
            wandb.finish()
        except:
            pass  # Ignore cleanup errors
    
    print(f"🎉 Training session complete!")

if __name__ == "__main__":
    main()