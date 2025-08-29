#!/usr/bin/env python3
"""
Viral Clip Generation - Practice Environment
This script helps you test and practice with all the installed libraries
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic library imports"""
    print("🔍 Testing basic library imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import scipy
        print("✅ SciPy imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_audio_libraries():
    """Test audio processing libraries"""
    print("\n🎵 Testing audio processing libraries...")
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
        
        import soundfile as sf
        print("✅ SoundFile imported successfully")
        
        import pydub
        print("✅ PyDub imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Audio library import error: {e}")
        return False

def test_video_libraries():
    """Test video processing libraries"""
    print("\n🎬 Testing video processing libraries...")
    
    try:
        import moviepy.editor as mp
        print("✅ MoviePy imported successfully")
        
        import ffmpeg
        print("✅ FFmpeg-Python imported successfully")
        
        from PIL import Image
        print("✅ Pillow (PIL) imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Video library import error: {e}")
        return False

def test_ai_libraries():
    """Test AI and machine learning libraries"""
    print("\n🤖 Testing AI and ML libraries...")
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
        
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        
        import torchaudio
        print("✅ TorchAudio imported successfully")
        
        import spacy
        print("✅ spaCy imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ AI library import error: {e}")
        return False

def test_web_frameworks():
    """Test web framework libraries"""
    print("\n🌐 Testing web framework libraries...")
    
    try:
        from flask import Flask
        print("✅ Flask imported successfully")
        
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
        
        import uvicorn
        print("✅ Uvicorn imported successfully")
        
        from flask_cors import CORS
        print("✅ Flask-CORS imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Web framework import error: {e}")
        return False

def test_utilities():
    """Test utility libraries"""
    print("\n🛠️ Testing utility libraries...")
    
    try:
        import requests
        print("✅ Requests imported successfully")
        
        import httpx
        print("✅ HTTPX imported successfully")
        
        import psutil
        print("✅ psutil imported successfully")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
        
        from pydantic import BaseModel
        print("✅ Pydantic imported successfully")
        
        from tqdm import tqdm
        print("✅ tqdm imported successfully")
        
        from tenacity import retry
        print("✅ Tenacity imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Utility library import error: {e}")
        return False

def test_task_management():
    """Test task management libraries"""
    print("\n⚡ Testing task management libraries...")
    
    try:
        from celery import Celery
        print("✅ Celery imported successfully")
        
        import redis
        print("✅ Redis imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Task management import error: {e}")
        return False

def create_sample_data():
    """Create sample data for practice"""
    print("\n📁 Creating sample data directory...")
    
    # Create directories
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    (sample_dir / "audio").mkdir(exist_ok=True)
    (sample_dir / "video").mkdir(exist_ok=True)
    (sample_dir / "images").mkdir(exist_ok=True)
    (sample_dir / "output").mkdir(exist_ok=True)
    
    print("✅ Sample data directories created")
    return sample_dir

def create_practice_scripts():
    """Create practice scripts for different libraries"""
    print("\n📝 Creating practice scripts...")
    
    scripts = {
        "01_audio_practice.py": """
# Audio Processing Practice
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def practice_audio_analysis():
    \"\"\"Practice audio analysis with librosa\"\"\"
    print("🎵 Audio Processing Practice")
    
    # Create a simple sine wave as example
    sample_rate = 22050
    duration = 5  # seconds
    frequency = 440  # Hz (A note)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Save audio file
    sf.write('sample_data/audio/sine_wave.wav', audio, sample_rate)
    print("✅ Created sample audio file: sine_wave.wav")
    
    # Load and analyze with librosa
    y, sr = librosa.load('sample_data/audio/sine_wave.wav')
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    print(f"✅ Extracted {mfcc.shape[1]} MFCC frames")
    print(f"✅ Extracted {len(spectral_centroids)} spectral centroid frames")
    
    return y, sr, mfcc, spectral_centroids

if __name__ == "__main__":
    practice_audio_analysis()
""",
        
        "02_video_practice.py": """
# Video Processing Practice
import numpy as np
from PIL import Image
import moviepy.editor as mp

def practice_video_creation():
    \"\"\"Practice video creation with MoviePy\"\"\"
    print("🎬 Video Processing Practice")
    
    # Create a simple colored frame
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Red channel
    
    # Save as image
    img = Image.fromarray(frame)
    img.save('sample_data/images/red_frame.png')
    print("✅ Created sample image: red_frame.png")
    
    # Create a simple video clip
    def make_frame(t):
        # Create a frame that changes color over time
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        intensity = int(255 * (t % 2) / 2)  # Oscillate between 0 and 255
        frame[:, :, 0] = intensity  # Red channel
        return frame
    
    clip = mp.VideoClip(make_frame, duration=3)
    clip.write_videofile('sample_data/video/color_animation.mp4', fps=24)
    print("✅ Created sample video: color_animation.mp4")
    
    return clip

if __name__ == "__main__":
    practice_video_creation()
""",
        
        "03_ai_practice.py": """
# AI and ML Practice
import torch
import torchaudio
import numpy as np

def practice_pytorch():
    \"\"\"Practice PyTorch operations\"\"\"
    print("🤖 AI/ML Practice")
    
    # Create a simple tensor
    x = torch.randn(3, 4)
    y = torch.randn(4, 3)
    
    # Matrix multiplication
    z = torch.mm(x, y)
    
    print(f"✅ Created tensor x: {x.shape}")
    print(f"✅ Created tensor y: {y.shape}")
    print(f"✅ Matrix multiplication result: {z.shape}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("✅ CUDA is available for GPU acceleration")
    else:
        print("ℹ️ CUDA not available, using CPU")
    
    return x, y, z

def practice_torchaudio():
    \"\"\"Practice TorchAudio operations\"\"\"
    print("🎵 TorchAudio Practice")
    
    # Create a simple audio tensor
    sample_rate = 22050
    duration = 2
    frequency = 440
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * frequency * t)
    
    # Add some noise
    noise = torch.randn_like(audio) * 0.1
    audio_with_noise = audio + noise
    
    print(f"✅ Created audio tensor: {audio_with_noise.shape}")
    print(f"✅ Sample rate: {sample_rate} Hz")
    
    return audio_with_noise, sample_rate

if __name__ == "__main__":
    practice_pytorch()
    practice_torchaudio()
""",
        
        "04_web_practice.py": """
# Web Framework Practice
from flask import Flask, jsonify
from fastapi import FastAPI
import uvicorn

def practice_flask():
    \"\"\"Practice Flask web framework\"\"\"
    print("🌐 Flask Practice")
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return jsonify({"message": "Hello from Flask!", "status": "success"})
    
    @app.route('/health')
    def health():
        return jsonify({"status": "healthy", "framework": "Flask"})
    
    print("✅ Flask app created with routes:")
    print("   - GET / (hello)")
    print("   - GET /health (health check)")
    
    return app

def practice_fastapi():
    \"\"\"Practice FastAPI web framework\"\"\"
    print("🚀 FastAPI Practice")
    
    app = FastAPI(title="Viral Clip API", version="1.0.0")
    
    @app.get("/")
    async def root():
        return {"message": "Hello from FastAPI!", "status": "success"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "framework": "FastAPI"}
    
    print("✅ FastAPI app created with routes:")
    print("   - GET / (root)")
    print("   - GET /health (health check)")
    
    return app

if __name__ == "__main__":
    flask_app = practice_flask()
    fastapi_app = practice_fastapi()
    
    print("\\nTo run Flask app:")
    print("flask_app.run(debug=True, port=5000)")
    
    print("\\nTo run FastAPI app:")
    print("uvicorn.run(fastapi_app, host='0.0.0.0', port=8000)")
""",
        
        "05_utilities_practice.py": """
# Utilities Practice
import requests
import httpx
import psutil
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
import time

def practice_requests():
    \"\"\"Practice HTTP requests\"\"\"
    print("🌍 HTTP Requests Practice")
    
    # Test with a public API
    try:
        response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
        data = response.json()
        print(f"✅ HTTP request successful: {response.status_code}")
        print(f"✅ Response data: {data['title'][:50]}...")
    except Exception as e:
        print(f"❌ HTTP request failed: {e}")
    
    return response

def practice_psutil():
    \"\"\"Practice system monitoring\"\"\"
    print("💻 System Monitoring Practice")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"✅ CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"✅ Memory Usage: {memory.percent}%")
    print(f"✅ Available Memory: {memory.available / (1024**3):.2f} GB")
    
    # Disk usage
    disk = psutil.disk_usage('/')
    print(f"✅ Disk Usage: {disk.percent}%")
    
    return cpu_percent, memory, disk

def practice_pydantic():
    \"\"\"Practice data validation with Pydantic\"\"\"
    print("📊 Data Validation Practice")
    
    class VideoClip(BaseModel):
        title: str
        duration: float
        format: str
        viral_score: float = 0.0
    
    # Valid data
    clip_data = {
        "title": "Amazing Viral Clip",
        "duration": 15.5,
        "format": "mp4",
        "viral_score": 8.7
    }
    
    try:
        clip = VideoClip(**clip_data)
        print(f"✅ Valid clip created: {clip.title}")
        print(f"✅ Duration: {clip.duration}s, Score: {clip.viral_score}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    return clip

def practice_tqdm():
    \"\"\"Practice progress bars\"\"\"
    print("📈 Progress Bar Practice")
    
    for i in tqdm(range(10), desc="Processing clips"):
        time.sleep(0.1)  # Simulate work
    
    print("✅ Progress bar demonstration completed")

if __name__ == "__main__":
    practice_requests()
    practice_psutil()
    practice_pydantic()
    practice_tqdm()
"""
    }
    
    for filename, content in scripts.items():
        with open(filename, 'w') as f:
            f.write(content.strip())
        print(f"✅ Created {filename}")
    
    return scripts

def main():
    """Main function to run all tests"""
    print("🚀 Viral Clip Generation - Practice Environment Setup")
    print("=" * 60)
    
    # Test all library imports
    tests = [
        test_basic_imports,
        test_audio_libraries,
        test_video_libraries,
        test_ai_libraries,
        test_web_frameworks,
        test_utilities,
        test_task_management
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    # Create sample data directory
    sample_dir = create_sample_data()
    
    # Create practice scripts
    scripts = create_practice_scripts()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All libraries imported successfully!")
        print("✅ Your environment is ready for viral clip generation practice!")
    else:
        print("⚠️ Some libraries failed to import. Check the errors above.")
    
    print("\n📚 Practice Scripts Created:")
    for script_name in scripts.keys():
        print(f"   - {script_name}")
    
    print("\n🎯 Next Steps:")
    print("1. Run individual practice scripts to learn each library")
    print("2. Start with: python 01_audio_practice.py")
    print("3. Explore the sample_data directory for your projects")
    print("4. Check the README.md for detailed usage examples")
    
    print(f"\n📁 Your workspace: {os.getcwd()}")
    print(f"📁 Sample data: {sample_dir.absolute()}")

if __name__ == "__main__":
    main()
