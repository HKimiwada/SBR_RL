# scaled_training.py - Full DGX-1 multi-core training with WandB monitoring
import os
import torch
import wandb
import numpy as np
from pathlib import Path
from sb3_stacking_environment import make_simple_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import time

class WandbCallback(BaseCallback):
    """Custom callback to log metrics to wandb"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
    
    def _on_step(self) -> bool:
        # Log training metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            if len(self.episode_rewards) > 0:
                wandb.log({
                    "train/episode_reward_mean": np.mean(self.episode_rewards[-100:]),
                    "train/episode_length_mean": np.mean(self.episode_lengths[-100:]),
                    "train/success_rate": np.mean(self.episode_successes[-100:]),
                    "train/total_timesteps": self.n_calls
                })
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Extract episode info from info buffer
        if hasattr(self.training_env, 'get_attr'):
            # Get episode info from all environments
            infos = self.training_env.get_attr('episode_returns')
            for env_infos in infos:
                if env_infos:
                    for info in env_infos:
                        if 'episode' in info:
                            self.episode_rewards.append(info['episode']['r'])
                            self.episode_lengths.append(info['episode']['l'])
                            # Extract success from final info if available
                            success = info.get('success', 0.0)
                            self.episode_successes.append(float(success))

def make_monitored_env(env_id=0, seed=0):
    """Create a monitored environment for multiprocessing"""
    def _init():
        # Set random seed for this process
        set_random_seed(seed + env_id)
        
        # Create environment
        env = make_simple_env('stack_3_bricks', max_episode_steps=1500)
        
        # Wrap with Monitor for episode statistics
        env = Monitor(env)
        
        # Reset with seed to initialize properly
        env.reset(seed=seed + env_id)
        
        return env
    
    return _init

class ScaledTrainingConfig:
    """Configuration for scaled training on DGX-1"""
    
    def __init__(self):
        # Environment settings
        self.task_variant = 'stack_3_bricks'
        self.max_episode_steps = 1500
        
        # Parallel training settings - utilize DGX-1's 40 cores
        self.n_envs = 32  # 32 parallel environments
        self.total_timesteps = 2_000_000  # 2M steps for thorough training
        
        # PPO hyperparameters - optimized for parallel training
        self.learning_rate = 3e-4
        self.n_steps = 2048  # Steps per environment before update
        self.batch_size = 256  # Larger batch size for stability
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        
        # Evaluation settings
        self.eval_freq = 25_000  # Evaluate every 25k steps
        self.n_eval_episodes = 10
        
        # Checkpointing
        self.checkpoint_freq = 100_000  # Save every 100k steps
        
        # Device settings
        self.device = 'cpu'  # CPU is better for MlpPolicy
        
        # Logging
        self.project_name = "stacking_robot_sb3"
        self.run_name = f"ppo_stack3bricks_{int(time.time())}"
        
        # Output directories
        self.output_dir = Path("./training_results")
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.videos_dir = self.output_dir / "videos"

def setup_directories(config):
    """Setup output directories"""
    config.output_dir.mkdir(exist_ok=True)
    config.models_dir.mkdir(exist_ok=True) 
    config.logs_dir.mkdir(exist_ok=True)
    config.videos_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {config.output_dir}")

def setup_wandb(config):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config={
            "algorithm": "PPO",
            "task": config.task_variant,
            "n_envs": config.n_envs,
            "total_timesteps": config.total_timesteps,
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "device": config.device
        },
        sync_tensorboard=True,  # Sync with tensorboard logs
        monitor_gym=True,       # Monitor gym environments
        save_code=True          # Save training code
    )
    print(f"🔍 WandB monitoring: {wandb.run.url}")

def create_vectorized_env(config):
    """Create vectorized environment for parallel training"""
    print(f"🏭 Creating {config.n_envs} parallel environments...")
    
    # Create environment functions
    env_fns = [make_monitored_env(env_id=i, seed=i) for i in range(config.n_envs)]
    
    # Use SubprocVecEnv for true parallelism
    if config.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    print(f"✅ Created vectorized environment with {config.n_envs} workers")
    return env

def create_eval_env():
    """Create evaluation environment"""
    eval_env_fn = make_monitored_env(env_id=999, seed=999)
    return DummyVecEnv([eval_env_fn])

def setup_callbacks(config, eval_env):
    """Setup training callbacks"""
    callbacks = []
    
    # WandB logging callback
    wandb_callback = WandbCallback(verbose=1)
    callbacks.append(wandb_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(config.models_dir / "best_model"),
        log_path=str(config.logs_dir),
        eval_freq=config.eval_freq // config.n_envs,  # Adjust for parallel envs
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq // config.n_envs,
        save_path=str(config.models_dir),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    return CallbackList(callbacks)

def create_model(env, config):
    """Create PPO model with optimized hyperparameters"""
    print(f"🤖 Creating PPO model...")
    print(f"   Device: {config.device}")
    print(f"   Parallel environments: {config.n_envs}")
    print(f"   Total timesteps: {config.total_timesteps:,}")
    
    # Optimized policy network architecture
    policy_kwargs = dict(
        net_arch=[256, 256, 128],  # Larger network for better learning
        activation_fn=torch.nn.ReLU,
    )
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        policy_kwargs=policy_kwargs,
        device=config.device,
        verbose=1,
        tensorboard_log=str(config.logs_dir),
    )
    
    print("✅ PPO model created successfully")
    return model

def log_system_info():
    """Log system information"""
    print(f"\n💻 System Information:")
    print(f"   CPU cores: {os.cpu_count()}")
    print(f"   Available memory: {torch.cuda.is_available() and torch.cuda.device_count()} GPUs")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Log to wandb
    wandb.log({
        "system/cpu_cores": os.cpu_count(),
        "system/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "system/pytorch_version": torch.__version__
    })

def main():
    """Main training function"""
    print("🚀 Starting Scaled DGX-1 Training")
    print("=" * 60)
    
    # Configuration
    config = ScaledTrainingConfig()
    
    # Setup
    setup_directories(config)
    setup_wandb(config)
    log_system_info()
    
    # Create environments
    train_env = create_vectorized_env(config)
    eval_env = create_eval_env()
    
    # Create model
    model = create_model(train_env, config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config, eval_env)
    
    # Training loop
    print(f"\n🎯 Starting training for {config.total_timesteps:,} steps...")
    print(f"   Expected duration: ~{config.total_timesteps // (config.n_envs * 1000):.0f} minutes")
    
    start_time = time.time()
    
    try:
        # Start training
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = config.models_dir / "ppo_final_model"
        model.save(str(final_model_path))
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"   Duration: {training_time/60:.1f} minutes")
        print(f"   Final model: {final_model_path}")
        
        # Log final stats
        wandb.log({
            "training/total_time_minutes": training_time / 60,
            "training/final_timesteps": config.total_timesteps,
            "training/completed": True
        })
        
        # Save training artifacts to wandb
        wandb.save(str(final_model_path) + ".zip")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted!")
        interrupted_model_path = config.models_dir / "ppo_interrupted_model"
        model.save(str(interrupted_model_path))
        print(f"   Model saved: {interrupted_model_path}")
        
    finally:
        # Cleanup
        train_env.close()
        eval_env.close()
        wandb.finish()
    
    print(f"\n🎉 Training session complete!")
    return model, config

if __name__ == "__main__":
    model, config = main()
