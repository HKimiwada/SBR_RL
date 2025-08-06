# fixed_stacking_environment.py
import os
os.environ['MUJOCO_GL'] = 'egl'  # Headless GPU rendering for DGX-1
os.environ['DISPLAY'] = ':0'     # Set display for X11 (if using X forwarding)

from dm_control import manipulation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time

class RGBStackingAlternative:
    """
    Fixed version with proper GPU rendering for DGX-1
    """
    
    def __init__(self, task_variant='stack_3_bricks', use_vision=False, render_size=(480, 640)):
        """
        Initialize the stacking environment with proper rendering
        
        Args:
            task_variant: 'stack_3_bricks', 'stack_2_bricks', etc.
            use_vision: If True, includes visual observations
            render_size: (height, width) for rendering
        """
        self.task_name = task_variant
        self.use_vision = use_vision
        self.render_height, self.render_width = render_size
        
        try:
            # Load environment - try features version first
            if use_vision:
                env_name = f"{task_variant}_vision"
            else:
                env_name = f"{task_variant}_features"
            
            self.env = manipulation.load(env_name)
            print(f"✅ Loaded: {env_name}")
            
        except Exception as e:
            # Fallback to basic task name
            print(f"⚠️  Failed to load {env_name}, trying {task_variant}")
            self.env = manipulation.load(task_variant)
            
        self._current_time_step = None
        
        # Test rendering capability
        self._test_rendering()
        
        print(f"🎮 RGBStackingAlternative Environment (GPU Rendering)")
        print(f"   Task: {self.task_name}")
        print(f"   Vision: {'Enabled' if use_vision else 'Disabled'}")
        print(f"   Render size: {render_size}")
    
    def _test_rendering(self):
        """Test if rendering works properly"""
        try:
            test_time_step = self.env.reset()
            test_img = self.env.physics.render(
                height=64, width=64, camera_id=0
            )
            if test_img is not None and test_img.size > 0:
                print("✅ GPU rendering working properly")
                self._rendering_works = True
            else:
                print("❌ Rendering returns empty image")
                self._rendering_works = False
        except Exception as e:
            print(f"❌ Rendering test failed: {e}")
            self._rendering_works = False
    
    def reset(self):
        """Reset environment"""
        self._current_time_step = self.env.reset()
        return self._get_observation()
    
    def step(self, action):
        """Step environment"""
        self._current_time_step = self.env.step(action)
        
        obs = self._get_observation()
        reward = float(self._current_time_step.reward)
        done = self._current_time_step.last()
        info = {
            'discount': self._current_time_step.discount,
            'success': reward > 0.5
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation with proper visual rendering"""
        if self._current_time_step is None:
            return None
        
        obs_dict = self._current_time_step.observation.copy()
        
        # Add visual observation if requested
        if self.use_vision and self._rendering_works:
            try:
                rgb_image = self.env.physics.render(
                    height=84, width=84, 
                    camera_id=0,
                    depth=False
                )
                obs_dict['rgb'] = rgb_image
                obs_dict['rgb_shape'] = rgb_image.shape
            except Exception as e:
                print(f"Warning: Visual observation failed: {e}")
        
        return obs_dict
    
    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        """
        Render environment with proper error handling
        
        Args:
            mode: 'rgb_array' for numpy array
            height, width: render resolution
            camera_id: which camera to use
        """
        if not self._rendering_works:
            print("❌ Rendering not available")
            return None
            
        h = height or self.render_height
        w = width or self.render_width
        
        try:
            rgb_image = self.env.physics.render(
                height=h, width=w, 
                camera_id=camera_id,
                depth=False
            )
            
            if rgb_image is None or rgb_image.size == 0:
                print("❌ Rendered image is empty")
                return None
                
            return rgb_image
            
        except Exception as e:
            print(f"❌ Rendering failed: {e}")
            return None
    
    def get_action_space(self):
        """Get action space"""
        spec = self.env.action_spec()
        return {
            'shape': spec.shape,
            'low': spec.minimum,
            'high': spec.maximum,
            'dtype': np.float32
        }
    
    def sample_random_action(self):
        """Sample random action"""
        spec = self.env.action_spec()
        return np.random.uniform(
            low=spec.minimum,
            high=spec.maximum,
            size=spec.shape
        ).astype(np.float32)
    
    def save_frame(self, filename, height=480, width=640):
        """
        Save current frame to file for transmission to Mac
        
        Args:
            filename: path to save image
            height, width: image resolution
        """
        img = self.render(height=height, width=width)
        if img is not None:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_bgr)
            return True
        return False
    
    def record_episode(self, output_dir, num_steps=100, fps=30):
        """
        Record a full episode as individual frames
        
        Args:
            output_dir: directory to save frames
            num_steps: max steps to record
            fps: target frames per second
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.reset()
        frames_saved = 0
        
        print(f"🎬 Recording episode to {output_dir}")
        
        for step in range(num_steps):
            # Take random action
            action = self.sample_random_action() * 0.1
            obs, reward, done, info = self.step(action)
            
            # Save frame
            frame_path = output_path / f"frame_{step:04d}.png"
            if self.save_frame(str(frame_path)):
                frames_saved += 1
                
                if step % 10 == 0:
                    print(f"   Saved frame {step}/{num_steps}")
            
            if done:
                print(f"   Episode ended at step {step}")
                break
        
        print(f"✅ Saved {frames_saved} frames to {output_dir}")
        
        # Create video from frames
        self._create_video_from_frames(output_dir, fps)
        
        return frames_saved
    
    def _create_video_from_frames(self, frame_dir, fps=30):
        """Create MP4 video from saved frames"""
        try:
            frame_path = Path(frame_dir)
            video_path = frame_path / "episode.mp4"
            
            # Get first frame to determine size
            first_frame = cv2.imread(str(frame_path / "frame_0000.png"))
            if first_frame is None:
                print("❌ No frames found for video creation")
                return
                
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            # Add frames to video
            frame_files = sorted(frame_path.glob("frame_*.png"))
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    out.write(frame)
            
            out.release()
            print(f"✅ Created video: {video_path}")
            
        except Exception as e:
            print(f"❌ Video creation failed: {e}")
    
    def close(self):
        """Close environment"""
        pass

# Test and demo functions
def test_gpu_rendering():
    """Test GPU rendering capabilities"""
    print("🔍 Testing GPU Rendering on DGX-1")
    print("=" * 50)
    
    # Test environment creation
    env = RGBStackingAlternative('stack_3_bricks', use_vision=True)
    
    # Test single frame rendering
    print("\n📸 Testing single frame rendering:")
    obs = env.reset()
    
    frame = env.render(height=240, width=320)
    if frame is not None:
        print(f"   ✅ Rendered frame: {frame.shape}")
        print(f"   ✅ Min/Max values: {frame.min()}/{frame.max()}")
        
        # Save test frame
        test_file = "/tmp/test_frame.png"
        if env.save_frame(test_file, height=240, width=320):
            print(f"   ✅ Saved test frame to: {test_file}")
    
    # Test episode recording
    print("\n🎬 Testing episode recording:")
    output_dir = "/tmp/episode_frames"
    frames_saved = env.record_episode(output_dir, num_steps=50, fps=15)
    
    print(f"\n✅ GPU rendering test complete!")
    print(f"   Frames saved: {frames_saved}")
    print(f"   Ready for transmission to Mac!")

if __name__ == "__main__":
    test_gpu_rendering()