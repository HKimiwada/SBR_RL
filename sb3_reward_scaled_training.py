# scaled_reward_training.py - Training with properly scaled rewards
import os
import torch
import wandb
import numpy as np
import time
from pathlib import Path
from sb3_reward_scaled_environment import make_scaled_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id=0, max_episode_steps=1500, reward_scale=1000.0):
    """Create a single reward-scaled environment"""
    def _init():
        set_random_seed(env_id)
        env = make_scaled_env(
            'stack_3_bricks', 
            max_episode_steps=max_episode_steps,
            reward_scale=reward_scale
        )
        env = Monitor(env)
        return env
    return _init

def create_env(n_envs=8, max_episode_steps=1500, reward_scale=1000.0):
    """Create vectorized environment with reward scaling"""
    print(f"🏭 Creating {n_envs} reward-scaled environments (scale={reward_scale}x)...")
    env_fns = [make_env(env_id=i, max_episode_steps=max_episode_steps, reward_scale=reward_scale) 
               for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    print(f"✅ Created {n_envs} environments with {reward_scale}x reward scaling")
    return env

def safe_get_episode_stats(ep_info_buffer, max_episodes=50):
    """Safely extract episode statistics from buffer"""
    if not ep_info_buffer or len(ep_info_buffer) == 0:
        return [], [], []
    
    episode_list = list(ep_info_buffer)
    recent_episodes = episode_list[-max_episodes:] if len(episode_list) > max_episodes else episode_list
    
    rewards = []
    lengths = []
    successes = []
    
    for ep in recent_episodes:
        if isinstance(ep, dict):
            if 'r' in ep:
                rewards.append(ep['r'])
            if 'l' in ep:
                lengths.append(ep['l'])
            if 'success' in ep:
                successes.append(float(ep['success']))
    
    return rewards, lengths, successes

def main():
    """Main training function with reward scaling"""
    print("🚀 Reward-Scaled Stacking Training")
    print("=" * 50)
    
    # ============ CONFIGURATION ============
    max_episode_steps = 1500      # Full episode length
    n_envs = 8                    # Parallel environments
    total_timesteps = 500_000     # Should learn much faster now
    reward_scale = 1000.0         # Scale tiny rewards by 1000x
    device = 'cpu'
    
    # Directories
    output_dir = Path("./training_results_scaled")
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    
    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    print(f"📊 Reward-Scaled Configuration:")
    print(f"   Episode length: {max_episode_steps} steps")
    print(f"   Environments: {n_envs}")
    print(f"   Total steps: {total_timesteps:,}")
    print(f"   Reward scale: {reward_scale}x (0.0004 → 0.4)")
    print(f"   Expected time: ~{total_timesteps // (n_envs * 1000)} minutes")
    
    # ============ SETUP WANDB ============
    run_name = f"scaled_reward_stacking_{int(time.time())}"
    
    wandb.init(
        project="stacking-robot-scaled-rewards",
        name=run_name,
        config={
            "algorithm": "PPO",
            "n_envs": n_envs,
            "total_timesteps": total_timesteps,
            "max_episode_steps": max_episode_steps,
            "reward_scale": reward_scale,
            "device": device
        }
    )
    print(f"🔍 WandB: {wandb.run.url}")
    
    # ============ CREATE ENVIRONMENTS ============
    train_env = create_env(n_envs, max_episode_steps, reward_scale)
    eval_env = create_env(1, max_episode_steps, reward_scale)
    
    # ============ CREATE MODEL ============
    print(f"🤖 Creating PPO model for scaled rewards...")
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,               # Standard PPO settings work now
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device=device,
        verbose=1,
        tensorboard_log=str(logs_dir),
    )
    
    print("✅ Model created for scaled reward environment")
    
    # ============ CALLBACKS ============
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=20_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=40_000 // n_envs,
        save_path=str(models_dir),
        name_prefix="scaled_checkpoint",
        verbose=1
    )
    
    # ============ TRAINING ============
    print(f"🎯 Starting scaled reward training...")
    start_time = time.time()
    
    steps_per_update = n_envs * 2048
    total_updates = total_timesteps // steps_per_update
    
    try:
        for update in range(total_updates):
            # Train for one update cycle
            model.learn(
                total_timesteps=steps_per_update,
                callback=[eval_callback, checkpoint_callback] if update % 5 == 0 else None,
                reset_num_timesteps=False,
                progress_bar=False
            )
            
            # Monitor progress
            if update % 3 == 0:
                current_timesteps = (update + 1) * steps_per_update
                progress = (update + 1) / total_updates
                
                # Get episode stats
                rewards, lengths, successes = safe_get_episode_stats(model.ep_info_buffer, max_episodes=30)
                
                if rewards and lengths:
                    mean_reward = np.mean(rewards)
                    mean_length = np.mean(lengths)
                    success_rate = np.mean(successes) if successes else 0.0
                    
                    # Convert back to original scale for logging
                    mean_original_reward = mean_reward / reward_scale
                    
                    # Log to WandB
                    wandb.log({
                        "train/timesteps": current_timesteps,
                        "train/progress": progress,
                        "train/mean_scaled_reward": mean_reward,
                        "train/mean_original_reward": mean_original_reward,
                        "train/mean_length": mean_length,
                        "train/success_rate": success_rate,
                        "train/episodes_completed": len(rewards)
                    })
                    
                    print(f"Update {update+1}/{total_updates} ({progress:.1%}) - "
                          f"Steps: {current_timesteps:,} - "
                          f"Scaled Reward: {mean_reward:.2f} - "
                          f"Original: {mean_original_reward:.6f} - "
                          f"Success: {success_rate:.1%} - "
                          f"Episodes: {len(rewards)}")
                else:
                    print(f"Update {update+1}/{total_updates} ({progress:.1%}) - "
                          f"Steps: {current_timesteps:,} - No completed episodes yet")
        
        # Final model save
        final_model_path = models_dir / "final_scaled_model"
        model.save(str(final_model_path))
        
        training_time = time.time() - start_time
        print(f"\n✅ Scaled reward training completed!")
        print(f"   Duration: {training_time/60:.1f} minutes")
        print(f"   Final model: {final_model_path}.zip")
        
        # Final stats
        rewards, lengths, successes = safe_get_episode_stats(model.ep_info_buffer, max_episodes=100)
        if rewards:
            final_mean_scaled = np.mean(rewards)
            final_mean_original = final_mean_scaled / reward_scale
            final_success_rate = np.mean(successes) if successes else 0.0
            final_mean_length = np.mean(lengths)
            
            wandb.log({
                "final/mean_scaled_reward": final_mean_scaled,
                "final/mean_original_reward": final_mean_original,
                "final/success_rate": final_success_rate,
                "final/mean_episode_length": final_mean_length,
                "final/training_time_minutes": training_time / 60,
                "final/total_episodes": len(rewards)
            })
            
            print(f"   Final stats:")
            print(f"     Mean scaled reward: {final_mean_scaled:.2f}")
            print(f"     Mean original reward: {final_mean_original:.6f}")
            print(f"     Success rate: {final_success_rate:.1%}")
            print(f"     Mean episode length: {final_mean_length:.0f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted!")
        interrupted_path = models_dir / "interrupted_scaled_model"
        model.save(str(interrupted_path))
        print(f"   Model saved: {interrupted_path}.zip")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        error_path = models_dir / "error_scaled_model"
        model.save(str(error_path))
        print(f"   Model saved: {error_path}.zip")
        
    finally:
        # Clean shutdown
        try:
            train_env.close()
            eval_env.close()
            wandb.finish()
        except:
            pass
    
    print(f"🎉 Scaled reward training complete!")
    print(f"💡 Your tiny 0.0004 rewards are now 0.4 - perfect for learning!")

if __name__ == "__main__":
    main()