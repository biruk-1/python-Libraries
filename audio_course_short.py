#!/usr/bin/env python3
"""
Short Audio Processing Course for Viral Clip Generation
Focused on the core libraries your PM specified
"""

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.signal as signal
import whisper
import os
from pathlib import Path

def lesson_1_numpy_basics():
    """Lesson 1: NumPy for Audio"""
    print("\n" + "="*50)
    print("LESSON 1: NumPy for Audio Processing")
    print("="*50)
    
    print("NumPy is the foundation for all audio processing.")
    
    # Create audio signal
    sample_rate = 22050
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Different audio signals
    sine_wave = np.sin(2 * np.pi * 440 * t)  # A note
    chord = (np.sin(2 * np.pi * 440 * t) +  # A
             np.sin(2 * np.pi * 554 * t) +  # C#
             np.sin(2 * np.pi * 659 * t))   # E
    chord = chord / 3
    
    print(f"✅ Created sine wave: {sine_wave.shape}")
    print(f"✅ Created chord: {chord.shape}")
    print(f"✅ Sample rate: {sample_rate} Hz")
    print(f"✅ Duration: {duration} seconds")
    
    # Save audio
    sample_dir = Path("sample_data/audio")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sf.write(sample_dir / "sine_wave.wav", sine_wave, sample_rate)
    sf.write(sample_dir / "chord.wav", chord, sample_rate)
    
    print("✅ Saved audio files")
    input("Press Enter to continue...")

def lesson_2_librosa_analysis():
    """Lesson 2: Librosa for Audio Analysis"""
    print("\n" + "="*50)
    print("LESSON 2: Librosa for Audio Analysis")
    print("="*50)
    
    print("Librosa is for audio analysis and feature extraction.")
    
    # Load audio
    audio_path = "sample_data/audio/chord.wav"
    y, sr = librosa.load(audio_path)
    
    print(f"✅ Loaded audio: {y.shape} samples at {sr} Hz")
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    print(f"✅ MFCC features: {mfcc.shape}")
    print(f"✅ Spectral centroids: {len(spectral_centroids)} frames")
    print(f"✅ Detected tempo: {tempo:.1f} BPM")
    print(f"✅ Number of beats: {len(beats)}")
    
    # Audio effects
    y_fast = librosa.effects.time_stretch(y, rate=1.5)
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    
    print(f"✅ Time-stretched audio: {len(y_fast) / sr:.2f} seconds")
    print(f"✅ Pitch-shifted audio: {y_pitch_up.shape}")
    
    input("Press Enter to continue...")

def lesson_3_soundfile_io():
    """Lesson 3: SoundFile for Audio I/O"""
    print("\n" + "="*50)
    print("LESSON 3: SoundFile for Audio I/O")
    print("="*50)
    
    print("SoundFile handles reading and writing audio files.")
    
    # Read audio
    audio_path = "sample_data/audio/chord.wav"
    data, samplerate = sf.read(audio_path)
    
    print(f"✅ Read audio: {data.shape} at {samplerate} Hz")
    
    # Create test signal
    t = np.linspace(0, 2, int(22050 * 2))
    test_signal = np.sin(2 * np.pi * 440 * t)
    
    # Save in different formats
    sample_dir = Path("sample_data/audio")
    sf.write(sample_dir / "test_16bit.wav", test_signal, 22050, subtype='PCM_16')
    sf.write(sample_dir / "test_float.wav", test_signal, 22050, subtype='FLOAT')
    
    print(f"✅ Saved 16-bit WAV")
    print(f"✅ Saved float WAV")
    
    # Get file info
    info = sf.info(audio_path)
    print(f"✅ File format: {info.format}")
    print(f"✅ Duration: {info.duration:.2f} seconds")
    print(f"✅ Channels: {info.channels}")
    
    input("Press Enter to continue...")

def lesson_4_pydub_manipulation():
    """Lesson 4: PyDub for Audio Manipulation"""
    print("\n" + "="*50)
    print("LESSON 4: PyDub for Audio Manipulation")
    print("="*50)
    
    print("PyDub is for high-level audio manipulation.")
    
    # Load audio
    audio_path = "sample_data/audio/chord.wav"
    audio = AudioSegment.from_wav(audio_path)
    
    print(f"✅ Loaded audio: {len(audio) / 1000:.2f} seconds")
    print(f"✅ Sample rate: {audio.frame_rate} Hz")
    print(f"✅ Channels: {audio.channels}")
    
    # Audio manipulation
    loud_audio = audio + 10  # Increase volume by 10dB
    quiet_audio = audio - 10  # Decrease volume by 10dB
    
    print(f"✅ Original dBFS: {audio.dBFS:.1f}")
    print(f"✅ Loud audio dBFS: {loud_audio.dBFS:.1f}")
    
    # Effects
    fade_in_out = audio.fade_in(500).fade_out(500)
    reversed_audio = audio.reverse()
    
    print(f"✅ Created fade effects")
    print(f"✅ Reversed audio: {len(reversed_audio) / 1000:.2f} seconds")
    
    # Format conversion
    audio.export("sample_data/audio/output_mp3.mp3", format="mp3")
    audio.export("sample_data/audio/output_ogg.ogg", format="ogg")
    
    print(f"✅ Exported to MP3")
    print(f"✅ Exported to OGG")
    
    input("Press Enter to continue...")

def lesson_5_scipy_processing():
    """Lesson 5: SciPy for Signal Processing"""
    print("\n" + "="*50)
    print("LESSON 5: SciPy for Signal Processing")
    print("="*50)
    
    print("SciPy provides advanced signal processing capabilities.")
    
    # Load audio
    data, sr = sf.read("sample_data/audio/chord.wav")
    
    print(f"✅ Loaded audio: {data.shape} at {sr} Hz")
    
    # Design filters
    cutoff = 1000  # Hz
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    
    # Low-pass filter
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_audio = signal.filtfilt(b, a, data)
    
    print(f"✅ Applied low-pass filter")
    print(f"✅ Original RMS: {np.sqrt(np.mean(data**2)):.3f}")
    print(f"✅ Filtered RMS: {np.sqrt(np.mean(filtered_audio**2)):.3f}")
    
    # Spectral analysis
    freqs, psd = signal.welch(data, sr, nperseg=1024)
    peak_freqs = freqs[np.argsort(psd)[-3:]]
    
    print(f"✅ PSD computed: {psd.shape}")
    print(f"✅ Top 3 frequencies: {peak_freqs}")
    
    # Echo effect
    delay_samples = int(0.1 * sr)  # 100ms delay
    echo_audio = np.zeros_like(data)
    echo_audio[delay_samples:] = data[:-delay_samples] * 0.5
    echo_audio += data
    
    print(f"✅ Created echo effect")
    
    input("Press Enter to continue...")

def lesson_6_whisper_transcription():
    """Lesson 6: Whisper for Speech Transcription"""
    print("\n" + "="*50)
    print("LESSON 6: Whisper for Speech Transcription")
    print("="*50)
    
    print("Whisper is OpenAI's speech-to-text transcription model.")
    
    # Load model
    print("Loading Whisper model (this may take a moment)...")
    model = whisper.load_model("base")
    print(f"✅ Model loaded: {model.name}")
    
    # Create sample speech-like audio
    t = np.linspace(0, 3, int(16000 * 3))
    speech_signal = (np.sin(2 * np.pi * 200 * t) +
                    np.sin(2 * np.pi * 400 * t) * 0.5 +
                    np.sin(2 * np.pi * 600 * t) * 0.3)
    
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    speech_signal *= envelope
    
    sf.write("sample_data/audio/sample_speech.wav", speech_signal, 16000)
    print(f"✅ Created sample speech audio")
    
    # Transcribe
    audio_path = "sample_data/audio/sample_speech.wav"
    result = model.transcribe(audio_path)
    
    print(f"✅ Transcription result:")
    print(f"   Text: {result['text']}")
    print(f"   Language: {result['language']}")
    print(f"   Segments: {len(result['segments'])}")
    
    # Different options
    result_detailed = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        word_timestamps=True
    )
    
    print(f"✅ Detailed transcription completed")
    print(f"   Full text: {result_detailed['text']}")
    
    input("Press Enter to continue...")

def final_assessment():
    """Final assessment"""
    print("\n" + "="*50)
    print("FINAL ASSESSMENT")
    print("="*50)
    
    questions = [
        {"q": "What library extracts MFCC features?", "a": "librosa", "points": 10},
        {"q": "Which library is best for audio format conversion?", "a": "pydub", "points": 10},
        {"q": "What function reads audio files in SoundFile?", "a": "sf.read()", "points": 10},
        {"q": "Which SciPy function designs Butterworth filters?", "a": "signal.butter()", "points": 10},
        {"q": "What Whisper function transcribes audio?", "a": "model.transcribe()", "points": 10}
    ]
    
    score = 0
    max_score = sum(q["points"] for q in questions)
    
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q['q']}")
        answer = input("Your answer: ").strip()
        print(f"Correct: {q['a']}")
        
        if q['a'].lower() in answer.lower() or answer.lower() in q['a'].lower():
            score += q["points"]
            print("✅ Correct!")
        else:
            print("❌ Incorrect")
    
    percentage = (score / max_score) * 100
    print(f"\nFinal Score: {score}/{max_score} ({percentage:.1f}%)")
    
    if percentage >= 80:
        print("🎉 Excellent! You've mastered audio processing!")
    elif percentage >= 60:
        print("👍 Good job! Solid understanding!")
    else:
        print("📚 Keep practicing!")

def main():
    """Run the complete course"""
    print("🎵 Audio Processing Course for Viral Clip Generation")
    print("="*60)
    print("Core libraries covered:")
    print("- NumPy: Numerical computing")
    print("- Librosa: Audio analysis")
    print("- SoundFile: Audio I/O")
    print("- PyDub: Audio manipulation")
    print("- SciPy: Signal processing")
    print("- Whisper: Speech transcription")
    print("="*60)
    
    lessons = [
        lesson_1_numpy_basics,
        lesson_2_librosa_analysis,
        lesson_3_soundfile_io,
        lesson_4_pydub_manipulation,
        lesson_5_scipy_processing,
        lesson_6_whisper_transcription
    ]
    
    for i, lesson in enumerate(lessons, 1):
        print(f"\nStarting Lesson {i}/6...")
        lesson()
    
    final_assessment()
    
    print("\n" + "="*60)
    print("🎉 COURSE COMPLETED!")
    print("="*60)
    print("You've learned:")
    print("✅ NumPy audio signal creation")
    print("✅ Librosa audio analysis")
    print("✅ SoundFile audio I/O")
    print("✅ PyDub audio manipulation")
    print("✅ SciPy signal processing")
    print("✅ Whisper transcription")
    print("\nReady for viral clip generation!")

if __name__ == "__main__":
    main()
