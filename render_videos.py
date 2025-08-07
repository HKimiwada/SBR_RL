# render_videos.py - Render videos of trained agent performance
import os
import cv2
import numpy as np
from pathlib import Path
from sb3_stacking_environment import make_simple_env
from stable_baselines3 import PPO
import time

class VideoRenderer:
    """Render videos of trained agent performance"""
    
    def __init__(self, model_path, output_dir="./videos"):
        """
        Initialize video renderer
        
        Args:
            model_path: Path to trained model (e.g., "ppo_stacking_50k")
            output_dir: Directory to save videos
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load trained model
        print(f"🤖 Loading trained model: {model_path}")
        self.model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Create environment for rendering
        self.env = make_simple_env('stack_3_bricks', max_episode_steps=250)
        print("✅ Environment created for rendering")
    
    def render_episode(self, episode_num=1, render_size=(640, 480), fps=30, deterministic=True):
        """
        Render a single episode and save as video
        
        Args:
            episode_num: Episode number for naming
            render_size: (width, height) for video
            fps: Frames per second
            deterministic: Use deterministic policy
            
        Returns:
            dict with episode stats
        """
        print(f"\n🎬 Recording Episode {episode_num}")
        
        # Create episode directory
        episode_dir = self.output_dir / f"episode_{episode_num:03d}"
        episode_dir.mkdir(exist_ok=True)
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Video writer setup
        video_path = episode_dir / f"episode_{episode_num:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path), 
            fourcc, 
            fps, 
            render_size
        )
        
        # Episode tracking
        episode_reward = 0
        episode_length = 0
        frames_saved = 0
        
        print(f"   🎥 Recording to: {video_path}")
        
        while True:
            # Get action from trained model
            action, _states = self.model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Render frame
            frame = self.env.render(mode='rgb_array')
            if frame is not None:
                # Resize if needed
                if frame.shape[:2] != (render_size[1], render_size[0]):
                    frame = cv2.resize(frame, render_size)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to video
                video_writer.write(frame_bgr)
                frames_saved += 1
                
                # Save individual frame (optional - for debugging)
                if episode_length % 10 == 0:  # Save every 10th frame
                    frame_path = episode_dir / f"frame_{episode_length:04d}.png"
                    cv2.imwrite(str(frame_path), frame_bgr)
            
            # Print progress
            if episode_length % 50 == 0:
                print(f"   Step {episode_length}: reward={reward:.4f}, total_reward={episode_reward:.3f}")
            
            # Check if episode ended
            if terminated or truncated:
                success = info.get('success', False)
                print(f"   🏁 Episode ended at step {episode_length}")
                print(f"   📊 Total reward: {episode_reward:.3f}")
                print(f"   🎯 Success: {success}")
                break
        
        # Close video writer
        video_writer.release()
        
        # Episode stats
        stats = {
            'episode': episode_num,
            'length': episode_length,
            'reward': episode_reward,
            'success': info.get('success', False),
            'frames_saved': frames_saved,
            'video_path': str(video_path)
        }
        
        print(f"   ✅ Video saved: {video_path} ({frames_saved} frames)")
        return stats
    
    def render_multiple_episodes(self, num_episodes=5, render_size=(640, 480), fps=30):
        """
        Render multiple episodes
        
        Args:
            num_episodes: Number of episodes to render
            render_size: Video dimensions
            fps: Frames per second
            
        Returns:
            List of episode statistics
        """
        print(f"🎬 Rendering {num_episodes} episodes")
        print("=" * 50)
        
        all_stats = []
        
        for episode in range(1, num_episodes + 1):
            stats = self.render_episode(
                episode_num=episode,
                render_size=render_size,
                fps=fps
            )
            all_stats.append(stats)
        
        # Summary statistics
        total_reward = sum(s['reward'] for s in all_stats)
        avg_reward = total_reward / len(all_stats)
        success_rate = sum(s['success'] for s in all_stats) / len(all_stats)
        avg_length = sum(s['length'] for s in all_stats) / len(all_stats)
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Average reward: {avg_reward:.3f}")
        print(f"   Success rate: {success_rate:.1%} ({sum(s['success'] for s in all_stats)}/{len(all_stats)})")
        print(f"   Average length: {avg_length:.1f}")
        print(f"   Total videos: {len(all_stats)}")
        
        # Create summary file
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Video Rendering Summary\n")
            f.write(f"=====================\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Episodes: {num_episodes}\n")
            f.write(f"Average Reward: {avg_reward:.3f}\n")
            f.write(f"Success Rate: {success_rate:.1%}\n")
            f.write(f"Average Length: {avg_length:.1f}\n\n")
            
            f.write("Episode Details:\n")
            for stats in all_stats:
                f.write(f"Episode {stats['episode']}: reward={stats['reward']:.3f}, "
                       f"length={stats['length']}, success={stats['success']}\n")
        
        print(f"   📄 Summary saved: {summary_path}")
        
        return all_stats
    
    def create_comparison_video(self, num_episodes=3, render_size=(640, 480)):
        """
        Create a comparison video showing multiple episodes side by side
        """
        print(f"🎞️  Creating comparison video...")
        
        # This is more advanced - for now, we'll just create individual videos
        # You can combine them later with video editing software
        return self.render_multiple_episodes(num_episodes, render_size)
    
    def close(self):
        """Close environment"""
        self.env.close()


def render_trained_model(model_path, num_episodes=3, video_quality="high"):
    """
    Main function to render videos of a trained model
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of episodes to record
        video_quality: "high" (640x480) or "medium" (480x360) or "low" (320x240)
    """
    print("🎬 Starting Video Rendering")
    print("=" * 50)
    
    # Set video quality
    quality_settings = {
        "high": (640, 480, 30),
        "medium": (480, 360, 25), 
        "low": (320, 240, 20)
    }
    
    width, height, fps = quality_settings.get(video_quality, quality_settings["high"])
    print(f"📺 Video settings: {width}x{height} @ {fps}fps")
    
    # Create renderer
    try:
        renderer = VideoRenderer(model_path, output_dir="./agent_videos")
        
        # Render episodes
        stats = renderer.render_multiple_episodes(
            num_episodes=num_episodes,
            render_size=(width, height),
            fps=fps
        )
        
        renderer.close()
        
        print(f"\n✅ Rendering complete!")
        print(f"📁 Videos saved in: ./agent_videos/")
        print(f"🎯 Ready to transfer to Mac!")
        
        return stats
        
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def transfer_instructions():
    """Print instructions for transferring videos to Mac"""
    print("\n" + "=" * 60)
    print("📦 HOW TO TRANSFER VIDEOS TO MAC")
    print("=" * 60)
    print()
    print("Option 1: SCP (Secure Copy)")
    print("   On your Mac terminal, run:")
    print("   scp -r username@dgx-1-ip:~/SBR_RL/agent_videos/ ~/Downloads/")
    print()
    print("Option 2: RSYNC")
    print("   rsync -avz username@dgx-1-ip:~/SBR_RL/agent_videos/ ~/Downloads/agent_videos/")
    print()
    print("Option 3: Zip and transfer")
    print("   On DGX-1:")
    print("   zip -r agent_videos.zip agent_videos/")
    print("   scp agent_videos.zip username@mac-ip:~/Downloads/")
    print()
    print("💡 Replace 'username@dgx-1-ip' with your actual DGX-1 details")
    print("💡 Videos are in MP4 format and should play on Mac")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python render_videos.py <model_path> [num_episodes] [quality]")
        print("Example: python render_videos.py ppo_stacking_50k 3 high")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    quality = sys.argv[3] if len(sys.argv) > 3 else "high"
    
    # Render videos
    stats = render_trained_model(model_path, num_episodes, quality)
    
    if stats:
        # Show transfer instructions
        transfer_instructions()