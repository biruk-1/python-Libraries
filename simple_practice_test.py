#!/usr/bin/env python3
"""
Viral Clip Generation - Simple Practice Test
This script tests all the installed libraries without emojis
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic library imports"""
    print("Testing basic library imports...")
    
    try:
        import numpy as np
        print("SUCCESS: NumPy imported")
        
        import pandas as pd
        print("SUCCESS: Pandas imported")
        
        import matplotlib.pyplot as plt
        print("SUCCESS: Matplotlib imported")
        
        import scipy
        print("SUCCESS: SciPy imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def test_audio_libraries():
    """Test audio processing libraries"""
    print("\nTesting audio processing libraries...")
    
    try:
        import librosa
        print("SUCCESS: Librosa imported")
        
        import soundfile as sf
        print("SUCCESS: SoundFile imported")
        
        import pydub
        print("SUCCESS: PyDub imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: Audio library import error: {e}")
        return False

def test_video_libraries():
    """Test video processing libraries"""
    print("\nTesting video processing libraries...")
    
    try:
        from moviepy.video import VideoClip
        print("SUCCESS: MoviePy video imported")
        
        import ffmpeg
        print("SUCCESS: FFmpeg-Python imported")
        
        from PIL import Image
        print("SUCCESS: Pillow (PIL) imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: Video library import error: {e}")
        return False

def test_ai_libraries():
    """Test AI and machine learning libraries"""
    print("\nTesting AI and ML libraries...")
    
    try:
        import openai
        print("SUCCESS: OpenAI imported")
        
        import torch
        print(f"SUCCESS: PyTorch imported (version: {torch.__version__})")
        
        import torchaudio
        print("SUCCESS: TorchAudio imported")
        
        import spacy
        print("SUCCESS: spaCy imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: AI library import error: {e}")
        return False

def test_web_frameworks():
    """Test web framework libraries"""
    print("\nTesting web framework libraries...")
    
    try:
        from flask import Flask
        print("SUCCESS: Flask imported")
        
        from fastapi import FastAPI
        print("SUCCESS: FastAPI imported")
        
        import uvicorn
        print("SUCCESS: Uvicorn imported")
        
        from flask_cors import CORS
        print("SUCCESS: Flask-CORS imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: Web framework import error: {e}")
        return False

def test_utilities():
    """Test utility libraries"""
    print("\nTesting utility libraries...")
    
    try:
        import requests
        print("SUCCESS: Requests imported")
        
        import httpx
        print("SUCCESS: HTTPX imported")
        
        import psutil
        print("SUCCESS: psutil imported")
        
        from dotenv import load_dotenv
        print("SUCCESS: python-dotenv imported")
        
        from pydantic import BaseModel
        print("SUCCESS: Pydantic imported")
        
        from tqdm import tqdm
        print("SUCCESS: tqdm imported")
        
        from tenacity import retry
        print("SUCCESS: Tenacity imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: Utility library import error: {e}")
        return False

def test_task_management():
    """Test task management libraries"""
    print("\nTesting task management libraries...")
    
    try:
        from celery import Celery
        print("SUCCESS: Celery imported")
        
        import redis
        print("SUCCESS: Redis imported")
        
        return True
    except ImportError as e:
        print(f"ERROR: Task management import error: {e}")
        return False

def create_sample_data():
    """Create sample data for practice"""
    print("\nCreating sample data directory...")
    
    # Create directories
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    (sample_dir / "audio").mkdir(exist_ok=True)
    (sample_dir / "video").mkdir(exist_ok=True)
    (sample_dir / "images").mkdir(exist_ok=True)
    (sample_dir / "output").mkdir(exist_ok=True)
    
    print("SUCCESS: Sample data directories created")
    return sample_dir

def main():
    """Main function to run all tests"""
    print("Viral Clip Generation - Practice Environment Test")
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
    
    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All libraries imported successfully!")
        print("Your environment is ready for viral clip generation practice!")
    else:
        print("WARNING: Some libraries failed to import. Check the errors above.")
    
    print("\nNext Steps:")
    print("1. Your environment is ready for practice")
    print("2. Explore the sample_data directory for your projects")
    print("3. Check the README.md for detailed usage examples")
    
    print(f"\nYour workspace: {os.getcwd()}")
    print(f"Sample data: {sample_dir.absolute()}")

if __name__ == "__main__":
    main()
