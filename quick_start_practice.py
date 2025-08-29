#!/usr/bin/env python3
"""
Quick Start Practice for Viral Clip Generation
This script demonstrates basic functionality of key libraries
"""

import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import requests
from pydantic import BaseModel
from tqdm import tqdm
import time

def practice_audio_processing():
    """Practice audio processing with librosa"""
    print("=== Audio Processing Practice ===")
    
    # Create a simple audio signal
    sample_rate = 22050
    duration = 3  # seconds
    frequency = 440  # Hz (A note)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Save audio file
    sf.write('sample_data/audio/test_tone.wav', audio, sample_rate)
    print("SUCCESS: Created test audio file")
    
    # Load and analyze with librosa
    y, sr = librosa.load('sample_data/audio/test_tone.wav')
    
    # Extract basic features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    print(f"SUCCESS: Extracted {mfcc.shape[1]} MFCC frames")
    print(f"SUCCESS: Extracted {len(spectral_centroids)} spectral centroid frames")
    
    return y, sr, mfcc

def practice_image_processing():
    """Practice image processing with Pillow"""
    print("\n=== Image Processing Practice ===")
    
    # Create a simple image
    width, height = 640, 480
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img_array[:, :, 0] = 255  # Red channel
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    img.save('sample_data/images/red_image.png')
    print("SUCCESS: Created red test image")
    
    # Resize the image
    resized_img = img.resize((320, 240))
    resized_img.save('sample_data/images/resized_image.png')
    print("SUCCESS: Created resized image")
    
    return img

def practice_http_requests():
    """Practice HTTP requests"""
    print("\n=== HTTP Requests Practice ===")
    
    try:
        # Test with a public API
        response = requests.get('https://jsonplaceholder.typicode.com/posts/1', timeout=5)
        data = response.json()
        print(f"SUCCESS: HTTP request successful (status: {response.status_code})")
        print(f"SUCCESS: Response title: {data['title'][:50]}...")
        return response
    except Exception as e:
        print(f"ERROR: HTTP request failed: {e}")
        return None

def practice_data_validation():
    """Practice data validation with Pydantic"""
    print("\n=== Data Validation Practice ===")
    
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
        print(f"SUCCESS: Valid clip created: {clip.title}")
        print(f"SUCCESS: Duration: {clip.duration}s, Score: {clip.viral_score}")
        return clip
    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        return None

def practice_progress_tracking():
    """Practice progress tracking with tqdm"""
    print("\n=== Progress Tracking Practice ===")
    
    for i in tqdm(range(5), desc="Processing clips"):
        time.sleep(0.5)  # Simulate work
    
    print("SUCCESS: Progress tracking demonstration completed")

def practice_numpy_operations():
    """Practice NumPy operations"""
    print("\n=== NumPy Operations Practice ===")
    
    # Create arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([10, 20, 30, 40, 50])
    
    # Basic operations
    sum_arr = arr1 + arr2
    product_arr = arr1 * arr2
    
    print(f"SUCCESS: Array 1: {arr1}")
    print(f"SUCCESS: Array 2: {arr2}")
    print(f"SUCCESS: Sum: {sum_arr}")
    print(f"SUCCESS: Product: {product_arr}")
    
    # Create a 2D array for image-like data
    image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"SUCCESS: Created image data array: {image_data.shape}")
    
    return arr1, arr2, image_data

def main():
    """Main practice function"""
    print("Viral Clip Generation - Quick Start Practice")
    print("=" * 50)
    
    # Run all practice functions
    practices = [
        practice_audio_processing,
        practice_image_processing,
        practice_http_requests,
        practice_data_validation,
        practice_progress_tracking,
        practice_numpy_operations
    ]
    
    results = {}
    for practice in practices:
        try:
            result = practice()
            results[practice.__name__] = result
        except Exception as e:
            print(f"ERROR in {practice.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print("PRACTICE COMPLETED!")
    print("\nWhat you've learned:")
    print("1. Audio processing with librosa and soundfile")
    print("2. Image processing with Pillow")
    print("3. HTTP requests with requests library")
    print("4. Data validation with Pydantic")
    print("5. Progress tracking with tqdm")
    print("6. Array operations with NumPy")
    
    print("\nNext steps:")
    print("1. Explore the sample_data directory")
    print("2. Try modifying the code to experiment")
    print("3. Check the README.md for more examples")
    print("4. Build your own viral clip generation project!")

if __name__ == "__main__":
    main()
