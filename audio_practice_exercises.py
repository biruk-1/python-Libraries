#!/usr/bin/env python3
"""
Audio Processing Practice Exercises
Quick exercises to reinforce your learning
"""

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.signal as signal
import whisper
from pathlib import Path

def exercise_1_numpy_audio():
    """Exercise 1: NumPy Audio Creation"""
    print("="*50)
    print("EXERCISE 1: NumPy Audio Creation")
    print("="*50)
    
    print("Create the following audio signals:")
    print("1. A 3-second sine wave at 880 Hz")
    print("2. A chord with frequencies 440, 554, 659 Hz")
    print("3. A signal with amplitude modulation")
    
    # Solutions
    sample_rate = 22050
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 1. 880 Hz sine wave
    sine_880 = np.sin(2 * np.pi * 880 * t)
    print(f"✅ Created 880 Hz sine wave: {sine_880.shape}")
    
    # 2. Chord
    chord = (np.sin(2 * np.pi * 440 * t) + 
             np.sin(2 * np.pi * 554 * t) + 
             np.sin(2 * np.pi * 659 * t)) / 3
    print(f"✅ Created chord: {chord.shape}")
    
    # 3. Amplitude modulation
    carrier = np.sin(2 * np.pi * 440 * t)
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    am_signal = carrier * modulator
    print(f"✅ Created AM signal: {am_signal.shape}")
    
    # Save files
    sample_dir = Path("sample_data/audio")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sf.write(sample_dir / "sine_880.wav", sine_880, sample_rate)
    sf.write(sample_dir / "chord_exercise.wav", chord, sample_rate)
    sf.write(sample_dir / "am_signal.wav", am_signal, sample_rate)
    print("✅ Saved all audio files")

def exercise_2_librosa_features():
    """Exercise 2: Librosa Feature Extraction"""
    print("\n" + "="*50)
    print("EXERCISE 2: Librosa Feature Extraction")
    print("="*50)
    
    print("Extract features from the chord audio:")
    print("1. MFCC features")
    print("2. Spectral rolloff")
    print("3. Zero crossing rate")
    
    # Load audio
    audio_path = "sample_data/audio/chord_exercise.wav"
    y, sr = librosa.load(audio_path)
    
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"✅ MFCC shape: {mfcc.shape}")
    
    # 2. Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    print(f"✅ Spectral rolloff: {len(rolloff)} frames")
    
    # 3. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    print(f"✅ Zero crossing rate: {len(zcr)} frames")
    
    # Additional features
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    print(f"✅ Spectral bandwidth: {len(spectral_bandwidth)} frames")

def exercise_3_soundfile_operations():
    """Exercise 3: SoundFile Operations"""
    print("\n" + "="*50)
    print("EXERCISE 3: SoundFile Operations")
    print("="*50)
    
    print("Perform these SoundFile operations:")
    print("1. Read audio and get metadata")
    print("2. Save in different formats")
    print("3. Create stereo audio")
    
    # 1. Read and get info
    audio_path = "sample_data/audio/chord_exercise.wav"
    data, sr = sf.read(audio_path)
    info = sf.info(audio_path)
    
    print(f"✅ Read audio: {data.shape} at {sr} Hz")
    print(f"✅ Duration: {info.duration:.2f} seconds")
    print(f"✅ Format: {info.format}")
    
    # 2. Save in different formats
    sample_dir = Path("sample_data/audio")
    sf.write(sample_dir / "output_24bit.wav", data, sr, subtype='PCM_24')
    sf.write(sample_dir / "output_32bit.wav", data, sr, subtype='PCM_32')
    print("✅ Saved 24-bit and 32-bit WAV files")
    
    # 3. Create stereo
    left_channel = data
    right_channel = data * 0.8  # Slightly quieter
    stereo_data = np.column_stack((left_channel, right_channel))
    sf.write(sample_dir / "stereo_output.wav", stereo_data, sr)
    print(f"✅ Created stereo audio: {stereo_data.shape}")

def exercise_4_pydub_effects():
    """Exercise 4: PyDub Effects"""
    print("\n" + "="*50)
    print("EXERCISE 4: PyDub Effects")
    print("="*50)
    
    print("Apply these PyDub effects:")
    print("1. Volume changes and normalization")
    print("2. Fade effects")
    print("3. Speed changes")
    
    # Load audio
    audio_path = "sample_data/audio/chord_exercise.wav"
    audio = AudioSegment.from_wav(audio_path)
    
    # 1. Volume changes
    loud_audio = audio + 15  # Increase by 15dB
    quiet_audio = audio - 15  # Decrease by 15dB
    normalized_audio = audio.normalize()
    
    print(f"✅ Original dBFS: {audio.dBFS:.1f}")
    print(f"✅ Loud audio dBFS: {loud_audio.dBFS:.1f}")
    print(f"✅ Normalized dBFS: {normalized_audio.dBFS:.1f}")
    
    # 2. Fade effects
    fade_in_out = audio.fade_in(1000).fade_out(1000)  # 1 second fades
    crossfade = audio.fade_in(500).fade_out(500)
    
    print("✅ Created fade effects")
    
    # 3. Speed changes
    fast_audio = audio.speedup(playback_speed=2.0)
    slow_audio = audio.speedup(playback_speed=0.5)
    
    print(f"✅ Fast audio: {len(fast_audio) / 1000:.2f} seconds")
    print(f"✅ Slow audio: {len(slow_audio) / 1000:.2f} seconds")
    
    # Export
    sample_dir = Path("sample_data/audio")
    loud_audio.export(sample_dir / "loud_audio.wav", format="wav")
    fade_in_out.export(sample_dir / "fade_audio.wav", format="wav")
    fast_audio.export(sample_dir / "fast_audio.wav", format="wav")
    print("✅ Exported processed audio files")

def exercise_5_scipy_filters():
    """Exercise 5: SciPy Filters"""
    print("\n" + "="*50)
    print("EXERCISE 5: SciPy Filters")
    print("="*50)
    
    print("Apply these SciPy filters:")
    print("1. Low-pass filter")
    print("2. High-pass filter")
    print("3. Band-pass filter")
    
    # Load audio
    data, sr = sf.read("sample_data/audio/chord_exercise.wav")
    nyquist = sr / 2
    
    # 1. Low-pass filter (cutoff at 500 Hz)
    low_cutoff = 500 / nyquist
    b_low, a_low = signal.butter(4, low_cutoff, btype='low')
    low_filtered = signal.filtfilt(b_low, a_low, data)
    
    # 2. High-pass filter (cutoff at 1000 Hz)
    high_cutoff = 1000 / nyquist
    b_high, a_high = signal.butter(4, high_cutoff, btype='high')
    high_filtered = signal.filtfilt(b_high, a_high, data)
    
    # 3. Band-pass filter (500-2000 Hz)
    band_low = 500 / nyquist
    band_high = 2000 / nyquist
    b_band, a_band = signal.butter(4, [band_low, band_high], btype='band')
    band_filtered = signal.filtfilt(b_band, a_band, data)
    
    print(f"✅ Applied low-pass filter (500 Hz)")
    print(f"✅ Applied high-pass filter (1000 Hz)")
    print(f"✅ Applied band-pass filter (500-2000 Hz)")
    
    # Save filtered audio
    sample_dir = Path("sample_data/audio")
    sf.write(sample_dir / "low_filtered.wav", low_filtered, sr)
    sf.write(sample_dir / "high_filtered.wav", high_filtered, sr)
    sf.write(sample_dir / "band_filtered.wav", band_filtered, sr)
    print("✅ Saved filtered audio files")

def exercise_6_whisper_basics():
    """Exercise 6: Whisper Basics"""
    print("\n" + "="*50)
    print("EXERCISE 6: Whisper Basics")
    print("="*50)
    
    print("Try these Whisper operations:")
    print("1. Load different model sizes")
    print("2. Transcribe with different options")
    print("3. Language detection")
    
    # Create a simple speech-like signal for testing
    t = np.linspace(0, 2, int(16000 * 2))
    speech_signal = (np.sin(2 * np.pi * 200 * t) +
                    np.sin(2 * np.pi * 400 * t) * 0.5)
    
    # Add some modulation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    speech_signal *= envelope
    
    # Save test audio
    sf.write("sample_data/audio/test_speech.wav", speech_signal, 16000)
    print("✅ Created test speech audio")
    
    # Load model (use base for speed)
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print(f"✅ Loaded model: {model.name}")
    
    # Transcribe
    audio_path = "sample_data/audio/test_speech.wav"
    result = model.transcribe(audio_path)
    
    print(f"✅ Transcription: {result['text']}")
    print(f"✅ Language: {result['language']}")
    
    # Try different options
    result_detailed = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        verbose=True
    )
    
    print(f"✅ Detailed transcription completed")
    print(f"✅ Segments: {len(result_detailed['segments'])}")

def run_all_exercises():
    """Run all exercises"""
    print("🎵 Audio Processing Practice Exercises")
    print("Complete these exercises to reinforce your learning!")
    
    exercises = [
        exercise_1_numpy_audio,
        exercise_2_librosa_features,
        exercise_3_soundfile_operations,
        exercise_4_pydub_effects,
        exercise_5_scipy_filters,
        exercise_6_whisper_basics
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n--- Exercise {i}/6 ---")
        exercise()
        input("\nPress Enter to continue to next exercise...")
    
    print("\n" + "="*60)
    print("🎉 All exercises completed!")
    print("="*60)
    print("\nYou've practiced:")
    print("✅ NumPy audio signal creation")
    print("✅ Librosa feature extraction")
    print("✅ SoundFile I/O operations")
    print("✅ PyDub audio effects")
    print("✅ SciPy signal filtering")
    print("✅ Whisper transcription")
    
    print("\nNext steps:")
    print("1. Try these exercises with your own audio files")
    print("2. Combine multiple libraries in one project")
    print("3. Apply to your viral clip generation work!")

if __name__ == "__main__":
    run_all_exercises()
