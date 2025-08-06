# test_dgx_mac_setup.py
"""
Test script to determine the best rendering method for your DGX-1 → Mac setup
Run this on your DGX-1 to test all rendering methods
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # GPU rendering

from fixed_stacking_environment import RGBStackingAlternative
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import sys

def test_1_basic_rendering():
    """Test 1: Basic GPU rendering capability"""
    print("\n" + "="*60)
    print("🔍 TEST 1: Basic GPU Rendering")
    print("="*60)
    
    try:
        env = RGBStackingAlternative('stack_3_bricks', use_vision=True)
        obs = env.reset()
        
        # Test single frame
        frame = env.render(height=240, width=320)
        
        if frame is not None and frame.size > 0:
            print("✅ GPU rendering works!")
            print(f"   Frame shape: {frame.shape}")
            print(f"   Data range: {frame.min()} - {frame.max()}")
            return True
        else:
            print("❌ GPU rendering failed - empty frame")
            return False
            
    except Exception as e:
        print(f"❌ GPU rendering failed: {e}")
        return False

def test_2_file_generation():
    """Test 2: File generation for transfer to Mac"""
    print("\n" + "="*60)
    print("📁 TEST 2: File Generation for Mac Transfer")
    print("="*60)
    
    try:
        env = RGBStackingAlternative('stack_3_bricks', use_vision=True)
        
        # Create test directory
        test_dir = Path("/tmp/dgx_mac_test")
        test_dir.mkdir(exist_ok=True)
        
        # Generate test content
        print("   Generating test frames and video...")
        frames_saved = env.record_episode(str(test_dir), num_steps=20, fps=10)
        
        # Check generated files
        frame_files = list(test_dir.glob("frame_*.png"))
        video_files = list(test_dir.glob("*.mp4"))
        
        print(f"✅ File generation successful!")
        print(f"   Frames saved: {frames_saved}")
        print(f"   Frame files: {len(frame_files)}")
        print(f"   Video files: {len(video_files)}")
        print(f"   📂 Test files in: {test_dir}")
        
        # Show transfer command
        print(f"\n📤 Transfer to Mac with:")
        print(f"   scp -r username@{get_hostname()}:{test_dir} ~/Desktop/")
        
        return True
        
    except Exception as e:
        print(f"❌ File generation failed: {e}")
        return False

def test_3_x11_capability():
    """Test 3: X11 forwarding capability"""
    print("\n" + "="*60)
    print("🖥️  TEST 3: X11 Forwarding Capability")
    print("="*60)
    
    # Check if DISPLAY is set
    display = os.environ.get('DISPLAY')
    if not display:
        print("❌ No DISPLAY variable set")
        print("   To enable: ssh -X username@dgx-1-server")
        return False
    
    print(f"✅ DISPLAY set to: {display}")
    
    # Test matplotlib with X11
    try:
        plt.ioff()  # Turn off interactive mode for test
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(np.random.rand(100, 100, 3))
        ax.set_title("X11 Test")
        
        # Try to save instead of show for testing
        test_path = "/tmp/x11_test.png"
        plt.savefig(test_path)
        plt.close()
        
        if Path(test_path).exists():
            print("✅ Matplotlib/X11 working!")
            print("   For live display, ensure you connected with: ssh -X")
            return True
        else:
            print("❌ Matplotlib save failed")
            return False
            
    except Exception as e:
        print(f"❌ X11 test failed: {e}")
        print("   Make sure you connected with: ssh -X username@dgx-1-server")
        return False

def test_4_jupyter_readiness():
    """Test 4: Jupyter notebook readiness"""
    print("\n" + "="*60)
    print("📓 TEST 4: Jupyter Notebook Readiness")
    print("="*60)
    
    try:
        import jupyter
        print("✅ Jupyter installed")
    except ImportError:
        print("❌ Jupyter not installed")
        print("   Install with: pip install jupyter")
        return False
    
    try:
        from IPython.display import Image, display
        print("✅ IPython display functions available")
    except ImportError:
        print("❌ IPython display not available")
        return False
    
    # Test image encoding
    try:
        env = RGBStackingAlternative('stack_3_bricks', use_vision=True)
        obs = env.reset()
        frame = env.render(height=200, width=300)
        
        if frame is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(frame)
            
            test_path = "/tmp/jupyter_test.png"
            pil_img.save(test_path)
            
            print("✅ Image processing for Jupyter works!")
            print(f"\n📓 Start Jupyter with:")
            print(f"   jupyter notebook --no-browser --port=8889 --ip=0.0.0.0")
            print(f"\n💻 Connect from Mac with:")
            print(f"   ssh -L 8889:localhost:8889 username@{get_hostname()}")
            print(f"   Then open: http://localhost:8889")
            
            return True
        else:
            print("❌ Frame rendering failed for Jupyter test")
            return False
            
    except Exception as e:
        print(f"❌ Jupyter test failed: {e}")
        return False

def test_5_performance_benchmark():
    """Test 5: Performance benchmark"""
    print("\n" + "="*60)
    print("⚡ TEST 5: Performance Benchmark")
    print("="*60)
    
    try:
        env = RGBStackingAlternative('stack_3_bricks', use_vision=True)
        env.reset()
        
        # Benchmark rendering speed
        sizes = [(240, 320), (480, 640), (720, 1280)]
        
        for height, width in sizes:
            start_time = time.time()
            num_frames = 10
            
            for _ in range(num_frames):
                frame = env.render(height=height, width=width)
                if frame is None:
                    break
            
            elapsed = time.time() - start_time
            fps = num_frames / elapsed if elapsed > 0 else 0
            
            print(f"   {width}x{height}: {fps:.1f} FPS")
        
        # Benchmark with actions
        print("\n   Testing with environment steps:")
        start_time = time.time()
        num_steps = 50
        
        for step in range(num_steps):
            action = env.sample_random_action() * 0.1
            obs, reward, done, info = env.step(action)
            frame = env.render(height=240, width=320)
            
            if done:
                env.reset()
        
        elapsed = time.time() - start_time
        fps = num_steps / elapsed
        print(f"   Full simulation: {fps:.1f} steps/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def get_hostname():
    """Get current hostname"""
    import socket
    return socket.gethostname()

def main():
    """Run all tests"""
    print("🚀 DGX-1 → Mac Rendering Setup Test")
    print("This will test all rendering methods for your setup")
    print(f"Hostname: {get_hostname()}")
    
    # Run all tests
    tests = [
        ("Basic GPU Rendering", test_1_basic_rendering),
        ("File Generation", test_2_file_generation), 
        ("X11 Forwarding", test_3_x11_capability),
        ("Jupyter Readiness", test_4_jupyter_readiness),
        ("Performance", test_5_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL" 
        print(f"   {test_name:20s}: {status}")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    
    if results.get("Basic GPU Rendering", False):
        print("✅ GPU rendering works!")
        
        if results.get("File Generation", False):
            print("🏆 BEST: Use file generation method")
            print("   - High quality, reliable")
            print("   - Generate videos on DGX-1, transfer to Mac")
            
        if results.get("Jupyter Readiness", False):
            print("🥈 GOOD: Use Jupyter notebook method")
            print("   - Interactive, good balance")
            print("   - Best for development")
            
        if results.get("X11 Forwarding", False):
            print("🥉 OK: X11 forwarding available")
            print("   - Real-time but may be slow")
            print("   - Good for quick tests")
    else:
        print("❌ GPU rendering not working - check your setup!")
        print("   - Ensure MUJOCO_GL=egl")
        print("   - Check dm_control installation")
    
    print(f"\n📂 Test files saved in: /tmp/dgx_mac_test/")
    print(f"🔄 Transfer command: scp -r username@{get_hostname()}:/tmp/dgx_mac_test ~/Desktop/")

if __name__ == "__main__":
    main()