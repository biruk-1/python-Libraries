#!/usr/bin/env python3
"""
Audio Processing Course for Viral Clip Generation
A comprehensive course covering the core audio libraries your PM specified
"""

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.signal as signal
import whisper
import os
import time
from pathlib import Path

class AudioProcessingCourse:
    def __init__(self):
        self.current_lesson = 1
        self.total_lessons = 6
        self.score = 0
        self.max_score = 0
        
    def print_header(self, title):
        """Print a formatted lesson header"""
        print("\n" + "="*60)
        print(f"LESSON {self.current_lesson}: {title}")
        print("="*60)
        
    def print_success(self, message):
        """Print a success message"""
        print(f"✅ {message}")
        
    def print_info(self, message):
        """Print an info message"""
        print(f"ℹ️ {message}")
        
    def print_exercise(self, message):
        """Print an exercise message"""
        print(f"🎯 EXERCISE: {message}")
        
    def wait_for_input(self):
        """Wait for user to press Enter to continue"""
        input("\nPress Enter to continue to the next section...")
        
    def create_sample_audio(self):
        """Create sample audio files for practice"""
        print("Creating sample audio files for practice...")
        
        # Create sample data directory
        sample_dir = Path("sample_data/audio")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple sine wave
        sample_rate = 22050
        duration = 3  # seconds
        frequency = 440  # Hz (A note)
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Save as WAV file
        sf.write(sample_dir / "sine_wave.wav", audio, sample_rate)
        
        # Create a more complex audio (chord)
        chord = (np.sin(2 * np.pi * 440 * t) +  # A
                np.sin(2 * np.pi * 554 * t) +  # C#
                np.sin(2 * np.pi * 659 * t))   # E
        chord = chord / 3  # Normalize
        
        sf.write(sample_dir / "chord.wav", chord, sample_rate)
        
        # Create audio with noise
        noise = np.random.normal(0, 0.1, len(audio))
        noisy_audio = audio + noise
        sf.write(sample_dir / "noisy_audio.wav", noisy_audio, sample_rate)
        
        self.print_success("Sample audio files created!")
        return sample_dir
        
    def lesson_1_numpy_audio_basics(self):
        """Lesson 1: NumPy for Audio Processing"""
        self.print_header("NumPy for Audio Processing")
        
        print("NumPy is the foundation for all audio processing. Let's learn how to work with audio data.")
        
        self.print_info("1. Creating audio signals with NumPy:")
        
        # Basic audio signal creation
        sample_rate = 22050
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create different types of audio signals
        sine_wave = np.sin(2 * np.pi * 440 * t)  # A note
        square_wave = np.sign(np.sin(2 * np.pi * 440 * t))
        sawtooth = 2 * (t * 440 - np.floor(t * 440 + 0.5))
        
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {duration} seconds")
        print(f"   Number of samples: {len(t)}")
        print(f"   Sine wave shape: {sine_wave.shape}")
        
        self.print_info("2. Audio signal properties:")
        print(f"   Sine wave amplitude range: [{sine_wave.min():.3f}, {sine_wave.max():.3f}]")
        print(f"   Sine wave mean: {sine_wave.mean():.3f}")
        print(f"   Sine wave RMS: {np.sqrt(np.mean(sine_wave**2)):.3f}")
        
        self.print_info("3. Audio signal manipulation:")
        # Amplitude scaling
        loud_audio = sine_wave * 2.0
        quiet_audio = sine_wave * 0.5
        
        # Frequency modulation
        fm_audio = np.sin(2 * np.pi * 440 * t + 0.5 * np.sin(2 * np.pi * 5 * t))
        
        print(f"   Loud audio max: {loud_audio.max():.3f}")
        print(f"   Quiet audio max: {quiet_audio.max():.3f}")
        
        self.print_info("4. Audio mixing and effects:")
        # Mix two signals
        mixed_audio = sine_wave + 0.5 * np.sin(2 * np.pi * 554 * t)  # A + C#
        
        # Add reverb-like effect (simple delay)
        delay_samples = int(0.1 * sample_rate)  # 100ms delay
        reverb_audio = np.zeros_like(sine_wave)
        reverb_audio[delay_samples:] = sine_wave[:-delay_samples] * 0.3
        reverb_audio += sine_wave
        
        print(f"   Mixed audio shape: {mixed_audio.shape}")
        print(f"   Reverb audio shape: {reverb_audio.shape}")
        
        self.print_exercise("Practice NumPy audio operations:")
        print("   - Create a chord with 3 different frequencies")
        print("   - Add noise to an audio signal")
        print("   - Create a fade-in effect")
        
        self.print_success("Lesson 1 completed! You understand NumPy audio basics.")
        self.wait_for_input()
        
    def lesson_2_librosa_audio_analysis(self):
        """Lesson 2: Librosa for Audio Analysis"""
        self.print_header("Librosa for Audio Analysis")
        
        print("Librosa is the go-to library for audio analysis and feature extraction.")
        
        # Load sample audio
        sample_dir = self.create_sample_audio()
        audio_path = sample_dir / "chord.wav"
        
        self.print_info("1. Loading and basic audio analysis:")
        
        # Load audio with librosa
        y, sr = librosa.load(str(audio_path))
        print(f"   Audio loaded: {y.shape} samples at {sr} Hz")
        print(f"   Duration: {len(y) / sr:.2f} seconds")
        print(f"   Amplitude range: [{y.min():.3f}, {y.max():.3f}]")
        
        self.print_info("2. Feature extraction:")
        
        # Extract various audio features
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"   MFCC shape: {mfcc.shape}")
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        print(f"   Spectral centroids: {len(spectral_centroids)} frames")
        
        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        print(f"   Chroma features shape: {chroma.shape}")
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        print(f"   Detected tempo: {tempo:.1f} BPM")
        print(f"   Number of beats: {len(beats)}")
        
        self.print_info("3. Advanced audio analysis:")
        
        # Harmonic and percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        print(f"   Harmonic component shape: {y_harmonic.shape}")
        print(f"   Percussive component shape: {y_percussive.shape}")
        
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        print(f"   Pitch tracking shape: {pitches.shape}")
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        print(f"   Number of onsets detected: {len(onset_times)}")
        
        self.print_info("4. Audio effects and transformations:")
        
        # Time stretching
        y_fast = librosa.effects.time_stretch(y, rate=1.5)
        y_slow = librosa.effects.time_stretch(y, rate=0.75)
        print(f"   Fast audio duration: {len(y_fast) / sr:.2f} seconds")
        print(f"   Slow audio duration: {len(y_slow) / sr:.2f} seconds")
        
        # Pitch shifting
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
        print(f"   Pitch-shifted audio shapes: {y_pitch_up.shape}, {y_pitch_down.shape}")
        
        self.print_exercise("Practice Librosa analysis:")
        print("   - Extract MFCC features from different audio files")
        print("   - Detect tempo and beats in music")
        print("   - Separate harmonic and percussive components")
        
        self.print_success("Lesson 2 completed! You can analyze audio with Librosa.")
        self.wait_for_input()
        
    def lesson_3_soundfile_io(self):
        """Lesson 3: SoundFile for Audio I/O"""
        self.print_header("SoundFile for Audio I/O")
        
        print("SoundFile is essential for reading and writing audio files efficiently.")
        
        self.print_info("1. Reading audio files:")
        
        # Read the sample audio
        sample_dir = Path("sample_data/audio")
        audio_path = sample_dir / "chord.wav"
        
        # Read audio data
        data, samplerate = sf.read(str(audio_path))
        print(f"   Audio data shape: {data.shape}")
        print(f"   Sample rate: {samplerate} Hz")
        print(f"   Data type: {data.dtype}")
        print(f"   Duration: {len(data) / samplerate:.2f} seconds")
        
        self.print_info("2. Writing audio files:")
        
        # Create different audio formats
        # Generate a test signal
        t = np.linspace(0, 2, int(22050 * 2))
        test_signal = np.sin(2 * np.pi * 440 * t)
        
        # Save in different formats
        sf.write(sample_dir / "test_16bit.wav", test_signal, 22050, subtype='PCM_16')
        sf.write(sample_dir / "test_24bit.wav", test_signal, 22050, subtype='PCM_24')
        sf.write(sample_dir / "test_float.wav", test_signal, 22050, subtype='FLOAT')
        
        print(f"   Saved 16-bit WAV: {sample_dir / 'test_16bit.wav'}")
        print(f"   Saved 24-bit WAV: {sample_dir / 'test_24bit.wav'}")
        print(f"   Saved float WAV: {sample_dir / 'test_float.wav'}")
        
        self.print_info("3. Audio format conversion:")
        
        # Read and convert formats
        data_16, sr_16 = sf.read(str(sample_dir / "test_16bit.wav"))
        data_24, sr_24 = sf.read(str(sample_dir / "test_24bit.wav"))
        data_float, sr_float = sf.read(str(sample_dir / "test_float.wav"))
        
        print(f"   16-bit data type: {data_16.dtype}")
        print(f"   24-bit data type: {data_24.dtype}")
        print(f"   Float data type: {data_float.dtype}")
        
        self.print_info("4. Multi-channel audio:")
        
        # Create stereo audio
        left_channel = np.sin(2 * np.pi * 440 * t)
        right_channel = np.sin(2 * np.pi * 554 * t)
        stereo_audio = np.column_stack((left_channel, right_channel))
        
        sf.write(sample_dir / "stereo_audio.wav", stereo_audio, 22050)
        print(f"   Stereo audio shape: {stereo_audio.shape}")
        
        # Read stereo audio
        stereo_data, stereo_sr = sf.read(str(sample_dir / "stereo_audio.wav"))
        print(f"   Loaded stereo shape: {stereo_data.shape}")
        
        self.print_info("5. Audio metadata and properties:")
        
        # Get audio file info
        info = sf.info(str(audio_path))
        print(f"   File format: {info.format}")
        print(f"   Sample rate: {info.samplerate} Hz")
        print(f"   Channels: {info.channels}")
        print(f"   Duration: {info.duration:.2f} seconds")
        print(f"   Subtype: {info.subtype}")
        
        self.print_exercise("Practice SoundFile operations:")
        print("   - Read audio files in different formats")
        print("   - Convert between audio formats")
        print("   - Create and save multi-channel audio")
        
        self.print_success("Lesson 3 completed! You can handle audio I/O with SoundFile.")
        self.wait_for_input()
        
    def lesson_4_pydub_audio_manipulation(self):
        """Lesson 4: PyDub for Audio Manipulation"""
        self.print_header("PyDub for Audio Manipulation")
        
        print("PyDub is excellent for high-level audio manipulation and format conversion.")
        
        self.print_info("1. Loading and basic operations:")
        
        # Load audio with PyDub
        sample_dir = Path("sample_data/audio")
        audio_path = sample_dir / "chord.wav"
        
        audio = AudioSegment.from_wav(str(audio_path))
        print(f"   Audio duration: {len(audio) / 1000:.2f} seconds")
        print(f"   Sample rate: {audio.frame_rate} Hz")
        print(f"   Channels: {audio.channels}")
        print(f"   Sample width: {audio.sample_width} bytes")
        
        self.print_info("2. Audio manipulation:")
        
        # Volume control
        loud_audio = audio + 10  # Increase volume by 10dB
        quiet_audio = audio - 10  # Decrease volume by 10dB
        
        print(f"   Original dBFS: {audio.dBFS:.1f}")
        print(f"   Loud audio dBFS: {loud_audio.dBFS:.1f}")
        print(f"   Quiet audio dBFS: {quiet_audio.dBFS:.1f}")
        
        # Speed and pitch manipulation
        fast_audio = audio.speedup(playback_speed=1.5)
        slow_audio = audio.speedup(playback_speed=0.75)
        
        print(f"   Fast audio duration: {len(fast_audio) / 1000:.2f} seconds")
        print(f"   Slow audio duration: {len(slow_audio) / 1000:.2f} seconds")
        
        self.print_info("3. Audio effects:")
        
        # Fade effects
        fade_in = audio.fade_in(1000)  # 1 second fade in
        fade_out = audio.fade_out(1000)  # 1 second fade out
        fade_in_out = audio.fade_in(500).fade_out(500)
        
        print(f"   Fade-in audio: {len(fade_in) / 1000:.2f} seconds")
        print(f"   Fade-out audio: {len(fade_out) / 1000:.2f} seconds")
        
        # Reverse audio
        reversed_audio = audio.reverse()
        print(f"   Reversed audio duration: {len(reversed_audio) / 1000:.2f} seconds")
        
        self.print_info("4. Audio concatenation and mixing:")
        
        # Create multiple audio segments
        segment1 = audio[:1000]  # First second
        segment2 = audio[1000:2000]  # Second second
        segment3 = audio[2000:3000]  # Third second
        
        # Concatenate
        concatenated = segment1 + segment2 + segment3
        print(f"   Concatenated duration: {len(concatenated) / 1000:.2f} seconds")
        
        # Mix audio (overlay)
        mixed = segment1.overlay(segment2)
        print(f"   Mixed audio duration: {len(mixed) / 1000:.2f} seconds")
        
        self.print_info("5. Format conversion:")
        
        # Export to different formats
        audio.export(sample_dir / "output_mp3.mp3", format="mp3")
        audio.export(sample_dir / "output_ogg.ogg", format="ogg")
        audio.export(sample_dir / "output_flac.flac", format="flac")
        
        print(f"   Exported to MP3: {sample_dir / 'output_mp3.mp3'}")
        print(f"   Exported to OGG: {sample_dir / 'output_ogg.ogg'}")
        print(f"   Exported to FLAC: {sample_dir / 'output_flac.flac'}")
        
        self.print_info("6. Audio analysis with PyDub:")
        
        # Get audio properties
        print(f"   Max possible amplitude: {audio.max_possible_amplitude}")
        print(f"   Max amplitude: {audio.max}")
        print(f"   Average amplitude: {audio.rms}")
        
        # Split audio into chunks
        chunk_length_ms = 1000  # 1 second chunks
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        print(f"   Number of chunks: {len(chunks)}")
        
        self.print_exercise("Practice PyDub manipulation:")
        print("   - Create audio with fade effects")
        print("   - Mix multiple audio tracks")
        print("   - Convert audio between formats")
        
        self.print_success("Lesson 4 completed! You can manipulate audio with PyDub.")
        self.wait_for_input()
        
    def lesson_5_scipy_signal_processing(self):
        """Lesson 5: SciPy for Signal Processing"""
        self.print_header("SciPy for Signal Processing")
        
        print("SciPy provides advanced signal processing capabilities for audio analysis.")
        
        self.print_info("1. Basic signal processing:")
        
        # Load sample audio
        sample_dir = Path("sample_data/audio")
        audio_path = sample_dir / "noisy_audio.wav"
        data, sr = sf.read(str(audio_path))
        
        print(f"   Audio data shape: {data.shape}")
        print(f"   Sample rate: {sr} Hz")
        
        self.print_info("2. Filtering:")
        
        # Design filters
        # Low-pass filter
        cutoff = 1000  # Hz
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, data)
        
        print(f"   Original audio RMS: {np.sqrt(np.mean(data**2)):.3f}")
        print(f"   Filtered audio RMS: {np.sqrt(np.mean(filtered_audio**2)):.3f}")
        
        # High-pass filter
        b_high, a_high = signal.butter(4, normal_cutoff, btype='high', analog=False)
        high_filtered = signal.filtfilt(b_high, a_high, data)
        
        # Band-pass filter
        low_cutoff = 500
        high_cutoff = 2000
        b_band, a_band = signal.butter(4, [low_cutoff/nyquist, high_cutoff/nyquist], btype='band', analog=False)
        band_filtered = signal.filtfilt(b_band, a_band, data)
        
        self.print_info("3. Spectral analysis:")
        
        # Compute power spectral density
        freqs, psd = signal.welch(data, sr, nperseg=1024)
        print(f"   PSD shape: {psd.shape}")
        print(f"   Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        
        # Find dominant frequencies
        peak_freqs = freqs[np.argsort(psd)[-5:]]
        print(f"   Top 5 frequencies: {peak_freqs}")
        
        self.print_info("4. Noise reduction:")
        
        # Simple noise reduction using spectral subtraction
        # Estimate noise from first 0.5 seconds
        noise_samples = int(0.5 * sr)
        noise_spectrum = np.mean(np.abs(np.fft.fft(data[:noise_samples])), axis=0)
        
        # Apply spectral subtraction
        signal_spectrum = np.abs(np.fft.fft(data))
        denoised_spectrum = signal_spectrum - 0.5 * noise_spectrum
        denoised_spectrum = np.maximum(denoised_spectrum, 0.1 * signal_spectrum)
        
        # Reconstruct signal
        denoised_audio = np.real(np.fft.ifft(denoised_spectrum))
        
        print(f"   Original SNR estimate: {np.sqrt(np.mean(data**2)) / np.sqrt(np.mean(data[:noise_samples]**2)):.2f}")
        print(f"   Denoised audio shape: {denoised_audio.shape}")
        
        self.print_info("5. Audio effects:")
        
        # Echo effect
        delay_samples = int(0.1 * sr)  # 100ms delay
        echo_audio = np.zeros_like(data)
        echo_audio[delay_samples:] = data[:-delay_samples] * 0.5
        echo_audio += data
        
        # Chorus effect (multiple delayed copies)
        chorus_audio = data.copy()
        for i in range(3):
            delay = int((0.02 + i * 0.01) * sr)  # 20-40ms delays
            delayed = np.zeros_like(data)
            delayed[delay:] = data[:-delay] * 0.3
            chorus_audio += delayed
        
        print(f"   Echo audio shape: {echo_audio.shape}")
        print(f"   Chorus audio shape: {chorus_audio.shape}")
        
        self.print_info("6. Advanced signal processing:")
        
        # Hilbert transform for envelope detection
        analytic_signal = signal.hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sr)
        
        print(f"   Amplitude envelope shape: {amplitude_envelope.shape}")
        print(f"   Instantaneous frequency shape: {instantaneous_frequency.shape}")
        
        self.print_exercise("Practice SciPy signal processing:")
        print("   - Design and apply different types of filters")
        print("   - Perform spectral analysis on audio")
        print("   - Implement noise reduction techniques")
        
        self.print_success("Lesson 5 completed! You can process signals with SciPy.")
        self.wait_for_input()
        
    def lesson_6_whisper_transcription(self):
        """Lesson 6: Whisper for Speech Transcription"""
        self.print_header("Whisper for Speech Transcription")
        
        print("Whisper is OpenAI's powerful speech-to-text transcription model.")
        
        self.print_info("1. Loading and basic usage:")
        
        # Load Whisper model
        print("Loading Whisper model (this may take a moment)...")
        model = whisper.load_model("base")  # Use base model for speed
        print(f"   Model loaded: {model.name}")
        
        self.print_info("2. Creating sample speech audio:")
        
        # For demonstration, we'll use our existing audio
        # In practice, you'd use actual speech audio
        sample_dir = Path("sample_data/audio")
        
        # Create a simple "speech-like" signal for demonstration
        t = np.linspace(0, 3, int(16000 * 3))  # 3 seconds at 16kHz
        # Simulate speech-like frequencies
        speech_signal = (np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
                        np.sin(2 * np.pi * 400 * t) * 0.5 +  # Harmonics
                        np.sin(2 * np.pi * 600 * t) * 0.3)
        
        # Add some modulation to simulate speech
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        speech_signal *= envelope
        
        # Save as WAV file
        sf.write(sample_dir / "sample_speech.wav", speech_signal, 16000)
        print(f"   Created sample speech audio: {sample_dir / 'sample_speech.wav'}")
        
        self.print_info("3. Basic transcription:")
        
        # Transcribe audio
        audio_path = sample_dir / "sample_speech.wav"
        result = model.transcribe(str(audio_path))
        
        print(f"   Transcription result:")
        print(f"   - Text: {result['text']}")
        print(f"   - Language: {result['language']}")
        print(f"   - Segments: {len(result['segments'])}")
        
        self.print_info("4. Advanced transcription options:")
        
        # Transcribe with different options
        result_detailed = model.transcribe(
            str(audio_path),
            language="en",  # Specify language
            task="transcribe",  # or "translate"
            verbose=True,
            word_timestamps=True
        )
        
        print(f"   Detailed transcription:")
        print(f"   - Full text: {result_detailed['text']}")
        print(f"   - Number of segments: {len(result_detailed['segments'])}")
        
        if result_detailed['segments']:
            first_segment = result_detailed['segments'][0]
            print(f"   - First segment:")
            print(f"     Start: {first_segment['start']:.2f}s")
            print(f"     End: {first_segment['end']:.2f}s")
            print(f"     Text: {first_segment['text']}")
        
        self.print_info("5. Different model sizes:")
        
        # Compare different model sizes
        model_sizes = ["tiny", "base", "small", "medium", "large"]
        
        print("   Available model sizes:")
        for size in model_sizes:
            print(f"     - {size}")
        
        print("   Note: Larger models are more accurate but slower")
        
        self.print_info("6. Batch processing:")
        
        # Process multiple audio files
        audio_files = [
            sample_dir / "sine_wave.wav",
            sample_dir / "chord.wav",
            sample_dir / "noisy_audio.wav"
        ]
        
        print("   Processing multiple audio files:")
        for audio_file in audio_files:
            try:
                result = model.transcribe(str(audio_file))
                print(f"     {audio_file.name}: {result['text'][:50]}...")
            except Exception as e:
                print(f"     {audio_file.name}: Error - {e}")
        
        self.print_info("7. Language detection and translation:")
        
        # Detect language
        result_lang = model.transcribe(str(audio_path), task="transcribe")
        detected_lang = result_lang['language']
        print(f"   Detected language: {detected_lang}")
        
        # Translate to English
        result_translate = model.transcribe(str(audio_path), task="translate")
        print(f"   Translation: {result_translate['text']}")
        
        self.print_exercise("Practice Whisper transcription:")
        print("   - Transcribe audio files with different models")
        print("   - Use different transcription options")
        print("   - Process multiple audio files in batch")
        
        self.print_success("Lesson 6 completed! You can transcribe speech with Whisper.")
        self.wait_for_input()
        
    def final_assessment(self):
        """Final assessment to test understanding"""
        self.print_header("Final Assessment")
        
        print("Let's test your audio processing knowledge!")
        
        questions = [
            {
                "question": "What library would you use to extract MFCC features from audio?",
                "answer": "librosa",
                "points": 10
            },
            {
                "question": "Which library is best for high-level audio manipulation like fading and format conversion?",
                "answer": "pydub",
                "points": 10
            },
            {
                "question": "What function in SoundFile would you use to read an audio file?",
                "answer": "sf.read()",
                "points": 10
            },
            {
                "question": "Which SciPy function would you use to design a Butterworth filter?",
                "answer": "signal.butter()",
                "points": 10
            },
            {
                "question": "What is the main function in Whisper for transcribing audio?",
                "answer": "model.transcribe()",
                "points": 10
            }
        ]
        
        self.max_score = sum(q["points"] for q in questions)
        
        print("Answer these questions to test your understanding:")
        print("(Type your answers, then press Enter to see the correct answer)")
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i} ({q['points']} points): {q['question']}")
            user_answer = input("Your answer: ").strip()
            print(f"Correct answer: {q['answer']}")
            
            if user_answer.lower() in q['answer'].lower() or q['answer'].lower() in user_answer.lower():
                self.score += q['points']
                self.print_success(f"Correct! +{q['points']} points")
            else:
                print(f"Not quite right. You got 0 points for this question.")
        
        percentage = (self.score / self.max_score) * 100
        print(f"\nFinal Score: {self.score}/{self.max_score} ({percentage:.1f}%)")
        
        if percentage >= 80:
            self.print_success("Excellent! You've mastered audio processing!")
        elif percentage >= 60:
            self.print_success("Good job! You have a solid understanding!")
        else:
            print("Keep practicing! Review the lessons and try again.")
            
    def run_course(self):
        """Run the complete audio processing course"""
        print("🎵 Welcome to the Audio Processing Course for Viral Clip Generation!")
        print("This course covers the core audio libraries your PM specified:")
        print("- NumPy: Numerical computing foundation")
        print("- Librosa: Audio signal processing and analysis")
        print("- SoundFile: Audio file I/O operations")
        print("- PyDub: Audio manipulation and format conversion")
        print("- SciPy: Scientific computing (signal processing)")
        print("- Whisper: OpenAI's speech-to-text transcription")
        
        lessons = [
            self.lesson_1_numpy_audio_basics,
            self.lesson_2_librosa_audio_analysis,
            self.lesson_3_soundfile_io,
            self.lesson_4_pydub_audio_manipulation,
            self.lesson_5_scipy_signal_processing,
            self.lesson_6_whisper_transcription
        ]
        
        for lesson in lessons:
            lesson()
            self.current_lesson += 1
            
        self.final_assessment()
        
        print("\n" + "="*60)
        print("🎉 CONGRATULATIONS! You've completed the Audio Processing course!")
        print("="*60)
        print("\nWhat you've learned:")
        print("✅ NumPy audio signal creation and manipulation")
        print("✅ Librosa audio analysis and feature extraction")
        print("✅ SoundFile audio I/O operations")
        print("✅ PyDub audio manipulation and effects")
        print("✅ SciPy signal processing and filtering")
        print("✅ Whisper speech transcription")
        
        print("\nNext steps:")
        print("1. Practice with your own audio files")
        print("2. Combine multiple libraries in your projects")
        print("3. Apply these skills to viral clip generation")
        print("4. Explore advanced features of each library")
        
        print(f"\nYour final score: {self.score}/{self.max_score}")

def main():
    """Main function to run the course"""
    course = AudioProcessingCourse()
    course.run_course()

if __name__ == "__main__":
    main()
