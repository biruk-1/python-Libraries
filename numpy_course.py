#!/usr/bin/env python3
"""
NumPy Course for Viral Clip Generation
A comprehensive course to learn NumPy from basics to advanced concepts
"""

import numpy as np
import time
from pathlib import Path

class NumPyCourse:
    def __init__(self):
        self.current_lesson = 1
        self.total_lessons = 8
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
        
    def lesson_1_basics(self):
        """Lesson 1: NumPy Basics - Creating Arrays"""
        self.print_header("NumPy Basics - Creating Arrays")
        
        print("NumPy is the foundation for scientific computing in Python.")
        print("It provides efficient array operations and mathematical functions.")
        
        self.print_info("Let's start by creating different types of arrays:")
        
        # 1. Creating arrays from lists
        print("\n1. Creating arrays from Python lists:")
        arr1 = np.array([1, 2, 3, 4, 5])
        print(f"   arr1 = np.array([1, 2, 3, 4, 5])")
        print(f"   Result: {arr1}")
        print(f"   Type: {type(arr1)}")
        print(f"   Shape: {arr1.shape}")
        print(f"   Data type: {arr1.dtype}")
        
        # 2. 2D arrays
        print("\n2. Creating 2D arrays:")
        arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(f"   arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])")
        print(f"   Result:\n{arr2d}")
        print(f"   Shape: {arr2d.shape}")
        
        # 3. Using np.zeros and np.ones
        print("\n3. Creating arrays with zeros and ones:")
        zeros_arr = np.zeros(5)
        ones_arr = np.ones((3, 3))
        print(f"   np.zeros(5): {zeros_arr}")
        print(f"   np.ones((3, 3)):\n{ones_arr}")
        
        # 4. Using np.arange and np.linspace
        print("\n4. Creating sequences:")
        range_arr = np.arange(0, 10, 2)  # start, stop, step
        linspace_arr = np.linspace(0, 1, 5)  # start, stop, num_points
        print(f"   np.arange(0, 10, 2): {range_arr}")
        print(f"   np.linspace(0, 1, 5): {linspace_arr}")
        
        # 5. Random arrays
        print("\n5. Creating random arrays:")
        random_arr = np.random.rand(3, 3)
        print(f"   np.random.rand(3, 3):\n{random_arr}")
        
        self.print_success("Lesson 1 completed! You now know how to create NumPy arrays.")
        self.wait_for_input()
        
    def lesson_2_indexing(self):
        """Lesson 2: Array Indexing and Slicing"""
        self.print_header("Array Indexing and Slicing")
        
        print("Indexing and slicing are crucial for accessing and manipulating array data.")
        
        # Create a sample array
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        print(f"Sample array:\n{arr}")
        print(f"Shape: {arr.shape}")
        
        self.print_info("1. Basic indexing:")
        print(f"   arr[0, 0] = {arr[0, 0]}  # First element")
        print(f"   arr[1, 2] = {arr[1, 2]}  # Second row, third column")
        print(f"   arr[-1, -1] = {arr[-1, -1]}  # Last element")
        
        self.print_info("2. Row and column indexing:")
        print(f"   arr[0] = {arr[0]}  # First row")
        print(f"   arr[:, 1] = {arr[:, 1]}  # Second column")
        
        self.print_info("3. Slicing:")
        print(f"   arr[0:2, 1:3] = \n{arr[0:2, 1:3]}  # First 2 rows, columns 1-2")
        print(f"   arr[::2, ::2] = \n{arr[::2, ::2]}  # Every 2nd row and column")
        
        self.print_info("4. Boolean indexing:")
        mask = arr > 5
        print(f"   arr > 5:\n{mask}")
        print(f"   arr[arr > 5] = {arr[arr > 5]}")
        
        self.print_info("5. Fancy indexing:")
        indices = [0, 2]
        print(f"   arr[indices, :] = \n{arr[indices, :]}  # Rows 0 and 2")
        
        self.print_exercise("Try accessing different elements:")
        print("   - Get the middle element of the array")
        print("   - Get all elements greater than 6")
        print("   - Get the last two columns")
        
        self.print_success("Lesson 2 completed! You can now access and slice array data.")
        self.wait_for_input()
        
    def lesson_3_operations(self):
        """Lesson 3: Array Operations"""
        self.print_header("Array Operations")
        
        print("NumPy provides efficient element-wise operations on arrays.")
        
        # Create sample arrays
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        print(f"a = {a}")
        print(f"b = {b}")
        
        self.print_info("1. Basic arithmetic operations:")
        print(f"   a + b = {a + b}")
        print(f"   a - b = {a - b}")
        print(f"   a * b = {a * b}")
        print(f"   a / b = {a / b}")
        print(f"   a ** 2 = {a ** 2}")
        
        self.print_info("2. Comparison operations:")
        print(f"   a > 2 = {a > 2}")
        print(f"   a == b = {a == b}")
        print(f"   a >= 3 = {a >= 3}")
        
        self.print_info("3. Mathematical functions:")
        print(f"   np.sqrt(a) = {np.sqrt(a)}")
        print(f"   np.sin(a) = {np.sin(a)}")
        print(f"   np.exp(a) = {np.exp(a)}")
        print(f"   np.log(a) = {np.log(a)}")
        
        self.print_info("4. Statistical operations:")
        print(f"   np.mean(a) = {np.mean(a)}")
        print(f"   np.std(a) = {np.std(a)}")
        print(f"   np.min(a) = {np.min(a)}")
        print(f"   np.max(a) = {np.max(a)}")
        print(f"   np.sum(a) = {np.sum(a)}")
        
        self.print_info("5. Broadcasting (different shapes):")
        c = np.array([[1, 2, 3], [4, 5, 6]])
        d = np.array([10, 20, 30])
        print(f"   c = \n{c}")
        print(f"   d = {d}")
        print(f"   c + d = \n{c + d}  # Broadcasting!")
        
        self.print_exercise("Practice operations:")
        print("   - Create two arrays and perform all basic operations")
        print("   - Calculate the mean and standard deviation of a random array")
        print("   - Use broadcasting to add a scalar to a 2D array")
        
        self.print_success("Lesson 3 completed! You can now perform array operations.")
        self.wait_for_input()
        
    def lesson_4_reshaping(self):
        """Lesson 4: Reshaping and Manipulating Arrays"""
        self.print_header("Reshaping and Manipulating Arrays")
        
        print("NumPy provides powerful tools to reshape and manipulate arrays.")
        
        # Create a sample array
        arr = np.arange(12)
        print(f"Original array: {arr}")
        print(f"Shape: {arr.shape}")
        
        self.print_info("1. Reshaping arrays:")
        arr_2d = arr.reshape(3, 4)
        print(f"   arr.reshape(3, 4):\n{arr_2d}")
        
        arr_3d = arr.reshape(2, 2, 3)
        print(f"   arr.reshape(2, 2, 3):\n{arr_3d}")
        
        self.print_info("2. Flattening arrays:")
        print(f"   arr_2d.flatten() = {arr_2d.flatten()}")
        print(f"   arr_2d.ravel() = {arr_2d.ravel()}")
        
        self.print_info("3. Transposing arrays:")
        print(f"   arr_2d.T:\n{arr_2d.T}")
        print(f"   np.transpose(arr_2d):\n{np.transpose(arr_2d)}")
        
        self.print_info("4. Adding and removing dimensions:")
        arr_1d = np.array([1, 2, 3, 4])
        print(f"   Original: {arr_1d}, shape: {arr_1d.shape}")
        
        arr_2d_new = arr_1d[:, np.newaxis]
        print(f"   arr_1d[:, np.newaxis]:\n{arr_2d_new}, shape: {arr_2d_new.shape}")
        
        arr_2d_alt = arr_1d.reshape(-1, 1)
        print(f"   arr_1d.reshape(-1, 1):\n{arr_2d_alt}")
        
        self.print_info("5. Concatenating arrays:")
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        print(f"   a = {a}, b = {b}")
        print(f"   np.concatenate([a, b]) = {np.concatenate([a, b])}")
        
        c = np.array([[1, 2], [3, 4]])
        d = np.array([[5, 6], [7, 8]])
        print(f"   c = \n{c}")
        print(f"   d = \n{d}")
        print(f"   np.vstack([c, d]) = \n{np.vstack([c, d])}  # Vertical stack")
        print(f"   np.hstack([c, d]) = \n{np.hstack([c, d])}  # Horizontal stack")
        
        self.print_exercise("Practice reshaping:")
        print("   - Create a 1D array and reshape it to 2D and 3D")
        print("   - Transpose a 2D array")
        print("   - Concatenate arrays vertically and horizontally")
        
        self.print_success("Lesson 4 completed! You can now reshape and manipulate arrays.")
        self.wait_for_input()
        
    def lesson_5_linear_algebra(self):
        """Lesson 5: Linear Algebra Operations"""
        self.print_header("Linear Algebra Operations")
        
        print("NumPy provides powerful linear algebra functions for matrix operations.")
        
        # Create sample matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        print(f"A = \n{A}")
        print(f"B = \n{B}")
        
        self.print_info("1. Matrix multiplication:")
        print(f"   A @ B = \n{A @ B}")
        print(f"   np.dot(A, B) = \n{np.dot(A, B)}")
        print(f"   np.matmul(A, B) = \n{np.matmul(A, B)}")
        
        self.print_info("2. Element-wise multiplication:")
        print(f"   A * B = \n{A * B}")
        
        self.print_info("3. Matrix properties:")
        print(f"   Determinant of A: {np.linalg.det(A)}")
        print(f"   Trace of A: {np.trace(A)}")
        print(f"   Rank of A: {np.linalg.matrix_rank(A)}")
        
        self.print_info("4. Matrix inverse and solving equations:")
        try:
            A_inv = np.linalg.inv(A)
            print(f"   Inverse of A:\n{A_inv}")
            print(f"   A @ A_inv = \n{A @ A_inv}")
        except np.linalg.LinAlgError:
            print("   Matrix A is not invertible")
        
        self.print_info("5. Eigenvalues and eigenvectors:")
        eigenvals, eigenvecs = np.linalg.eig(A)
        print(f"   Eigenvalues: {eigenvals}")
        print(f"   Eigenvectors:\n{eigenvecs}")
        
        self.print_info("6. Solving linear equations Ax = b:")
        b = np.array([5, 11])
        x = np.linalg.solve(A, b)
        print(f"   b = {b}")
        print(f"   Solution x = {x}")
        print(f"   Verification: A @ x = {A @ x}")
        
        self.print_info("7. Singular Value Decomposition (SVD):")
        U, s, Vt = np.linalg.svd(A)
        print(f"   U:\n{U}")
        print(f"   Singular values: {s}")
        print(f"   V^T:\n{Vt}")
        
        self.print_exercise("Practice linear algebra:")
        print("   - Create two matrices and multiply them")
        print("   - Calculate the determinant and inverse of a matrix")
        print("   - Solve a system of linear equations")
        
        self.print_success("Lesson 5 completed! You can now perform linear algebra operations.")
        self.wait_for_input()
        
    def lesson_6_random_numbers(self):
        """Lesson 6: Random Number Generation"""
        self.print_header("Random Number Generation")
        
        print("NumPy provides comprehensive random number generation capabilities.")
        
        self.print_info("1. Setting random seed for reproducibility:")
        np.random.seed(42)
        print("   np.random.seed(42)  # Set seed for reproducible results")
        
        self.print_info("2. Basic random number generation:")
        print(f"   np.random.rand(5) = {np.random.rand(5)}  # Uniform [0, 1)")
        print(f"   np.random.randn(5) = {np.random.randn(5)}  # Standard normal")
        print(f"   np.random.randint(1, 10, 5) = {np.random.randint(1, 10, 5)}  # Integers")
        
        self.print_info("3. Random arrays with specific shapes:")
        print(f"   np.random.rand(3, 3):\n{np.random.rand(3, 3)}")
        print(f"   np.random.randn(2, 4):\n{np.random.randn(2, 4)}")
        
        self.print_info("4. Probability distributions:")
        # Normal distribution
        normal_data = np.random.normal(0, 1, 1000)
        print(f"   Normal distribution (mean=0, std=1): mean={np.mean(normal_data):.3f}, std={np.std(normal_data):.3f}")
        
        # Uniform distribution
        uniform_data = np.random.uniform(0, 10, 1000)
        print(f"   Uniform distribution [0, 10): min={np.min(uniform_data):.3f}, max={np.max(uniform_data):.3f}")
        
        # Exponential distribution
        exp_data = np.random.exponential(1, 1000)
        print(f"   Exponential distribution (scale=1): mean={np.mean(exp_data):.3f}")
        
        self.print_info("5. Random choice and shuffling:")
        choices = ['apple', 'banana', 'cherry', 'date']
        print(f"   Choices: {choices}")
        print(f"   np.random.choice(choices, 3) = {np.random.choice(choices, 3)}")
        print(f"   np.random.choice(choices, 3, p=[0.5, 0.3, 0.1, 0.1]) = {np.random.choice(choices, 3, p=[0.5, 0.3, 0.1, 0.1])}")
        
        arr = np.arange(10)
        np.random.shuffle(arr)
        print(f"   Shuffled array: {arr}")
        
        self.print_info("6. Random sampling:")
        population = np.arange(100)
        sample = np.random.choice(population, 10, replace=False)
        print(f"   Random sample without replacement: {sample}")
        
        self.print_exercise("Practice random number generation:")
        print("   - Generate random numbers from different distributions")
        print("   - Create a random matrix and calculate its properties")
        print("   - Simulate coin flips or dice rolls")
        
        self.print_success("Lesson 6 completed! You can now generate random numbers.")
        self.wait_for_input()
        
    def lesson_7_performance(self):
        """Lesson 7: Performance and Optimization"""
        self.print_header("Performance and Optimization")
        
        print("NumPy is designed for high performance. Let's compare with Python lists.")
        
        self.print_info("1. Speed comparison: NumPy vs Python lists")
        
        # Create large arrays
        size = 1000000
        python_list = list(range(size))
        numpy_array = np.arange(size)
        
        # Time operations
        start_time = time.time()
        python_result = [x * 2 for x in python_list]
        python_time = time.time() - start_time
        
        start_time = time.time()
        numpy_result = numpy_array * 2
        numpy_time = time.time() - start_time
        
        print(f"   Python list multiplication: {python_time:.4f} seconds")
        print(f"   NumPy array multiplication: {numpy_time:.4f} seconds")
        print(f"   Speedup: {python_time/numpy_time:.1f}x faster!")
        
        self.print_info("2. Memory efficiency:")
        import sys
        print(f"   Python list memory: {sys.getsizeof(python_list)} bytes")
        print(f"   NumPy array memory: {numpy_array.nbytes} bytes")
        print(f"   Memory efficiency: {sys.getsizeof(python_list)/numpy_array.nbytes:.1f}x more efficient!")
        
        self.print_info("3. Vectorized operations:")
        # Create sample data
        data = np.random.randn(10000)
        
        # Non-vectorized (slow)
        start_time = time.time()
        result_slow = np.array([np.sin(x) for x in data])
        slow_time = time.time() - start_time
        
        # Vectorized (fast)
        start_time = time.time()
        result_fast = np.sin(data)
        fast_time = time.time() - start_time
        
        print(f"   Non-vectorized sin: {slow_time:.4f} seconds")
        print(f"   Vectorized sin: {fast_time:.4f} seconds")
        print(f"   Speedup: {slow_time/fast_time:.1f}x faster!")
        
        self.print_info("4. Broadcasting for efficiency:")
        # Create matrices
        A = np.random.randn(1000, 1000)
        B = np.random.randn(1000, 1000)
        
        # Element-wise operations
        start_time = time.time()
        C = A + B
        add_time = time.time() - start_time
        
        start_time = time.time()
        D = A * B
        mul_time = time.time() - start_time
        
        print(f"   Matrix addition: {add_time:.4f} seconds")
        print(f"   Matrix multiplication: {mul_time:.4f} seconds")
        
        self.print_info("5. Memory views and copying:")
        original = np.array([1, 2, 3, 4, 5])
        view = original[1:4]  # Creates a view (no copy)
        copy = original[1:4].copy()  # Creates a copy
        
        print(f"   Original: {original}")
        print(f"   View: {view}")
        print(f"   Copy: {copy}")
        
        # Modify view
        view[0] = 99
        print(f"   After modifying view: original={original}, view={view}")
        
        # Modify copy
        copy[0] = 88
        print(f"   After modifying copy: original={original}, copy={copy}")
        
        self.print_exercise("Performance practice:")
        print("   - Compare the speed of different operations")
        print("   - Use vectorized operations instead of loops")
        print("   - Understand when to use views vs copies")
        
        self.print_success("Lesson 7 completed! You understand NumPy performance.")
        self.wait_for_input()
        
    def lesson_8_practical_applications(self):
        """Lesson 8: Practical Applications for Viral Clip Generation"""
        self.print_header("Practical Applications for Viral Clip Generation")
        
        print("Let's apply NumPy to real scenarios you'll encounter in viral clip generation.")
        
        self.print_info("1. Image data representation:")
        # Create a simple grayscale image
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        print(f"   Grayscale image shape: {image.shape}")
        print(f"   Image data type: {image.dtype}")
        print(f"   Min/Max pixel values: {image.min()}/{image.max()}")
        
        # Create a color image (RGB)
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        print(f"   Color image shape: {color_image.shape}")
        print(f"   Red channel mean: {color_image[:, :, 0].mean():.1f}")
        print(f"   Green channel mean: {color_image[:, :, 1].mean():.1f}")
        print(f"   Blue channel mean: {color_image[:, :, 2].mean():.1f}")
        
        self.print_info("2. Audio signal processing:")
        # Create a simple audio signal
        sample_rate = 22050
        duration = 1  # second
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # Hz (A note)
        audio_signal = np.sin(2 * np.pi * frequency * t)
        
        print(f"   Audio signal shape: {audio_signal.shape}")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {duration} second")
        print(f"   Frequency: {frequency} Hz")
        print(f"   Signal amplitude range: [{audio_signal.min():.3f}, {audio_signal.max():.3f}]")
        
        self.print_info("3. Video frame processing:")
        # Simulate video frames
        num_frames = 30
        frame_height, frame_width = 480, 640
        video_frames = np.random.randint(0, 256, (num_frames, frame_height, frame_width, 3), dtype=np.uint8)
        
        print(f"   Video shape: {video_frames.shape}")
        print(f"   Number of frames: {num_frames}")
        print(f"   Frame resolution: {frame_width}x{frame_height}")
        print(f"   Total video size: {video_frames.nbytes / (1024*1024):.1f} MB")
        
        self.print_info("4. Feature extraction for viral prediction:")
        # Simulate video features
        num_videos = 100
        features = np.random.randn(num_videos, 10)  # 10 features per video
        
        # Calculate viral score based on features
        weights = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02])
        viral_scores = np.dot(features, weights)
        
        print(f"   Number of videos: {num_videos}")
        print(f"   Features per video: {features.shape[1]}")
        print(f"   Viral scores range: [{viral_scores.min():.3f}, {viral_scores.max():.3f}]")
        print(f"   Average viral score: {viral_scores.mean():.3f}")
        
        # Find top viral videos
        top_indices = np.argsort(viral_scores)[-5:]
        print(f"   Top 5 viral video indices: {top_indices}")
        print(f"   Top 5 viral scores: {viral_scores[top_indices]}")
        
        self.print_info("5. Data preprocessing and normalization:")
        # Normalize features
        features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
        print(f"   Original features mean: {features.mean():.3f}, std: {features.std():.3f}")
        print(f"   Normalized features mean: {features_normalized.mean():.3f}, std: {features_normalized.std():.3f}")
        
        self.print_info("6. Batch processing for efficiency:")
        batch_size = 10
        num_batches = num_videos // batch_size
        
        print(f"   Processing {num_videos} videos in batches of {batch_size}")
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = features[start_idx:end_idx]
            batch_scores = np.dot(batch, weights)
            print(f"   Batch {i+1}: mean score = {batch_scores.mean():.3f}")
        
        self.print_exercise("Viral clip generation practice:")
        print("   - Create a function to extract basic features from video frames")
        print("   - Implement a simple viral score calculation")
        print("   - Process audio data to extract frequency features")
        print("   - Create a batch processing pipeline for multiple videos")
        
        self.print_success("Lesson 8 completed! You can now apply NumPy to viral clip generation!")
        self.wait_for_input()
        
    def final_assessment(self):
        """Final assessment to test understanding"""
        self.print_header("Final Assessment")
        
        print("Let's test your NumPy knowledge with some practical exercises!")
        
        questions = [
            {
                "question": "Create a 3x3 identity matrix using NumPy",
                "answer": "np.eye(3) or np.identity(3)",
                "points": 10
            },
            {
                "question": "How do you find the maximum value in a 2D array?",
                "answer": "np.max(array) or array.max()",
                "points": 10
            },
            {
                "question": "What's the difference between np.array([1,2,3]) and np.array([[1,2,3]])?",
                "answer": "First is 1D, second is 2D (row vector)",
                "points": 15
            },
            {
                "question": "How do you reshape an array of 12 elements into a 3x4 matrix?",
                "answer": "array.reshape(3, 4) or array.reshape(-1, 4)",
                "points": 10
            },
            {
                "question": "What does broadcasting mean in NumPy?",
                "answer": "Automatic expansion of arrays with different shapes for operations",
                "points": 15
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
            self.print_success("Excellent! You've mastered NumPy basics!")
        elif percentage >= 60:
            self.print_success("Good job! You have a solid understanding of NumPy!")
        else:
            print("Keep practicing! Review the lessons and try again.")
            
    def run_course(self):
        """Run the complete NumPy course"""
        print("🚀 Welcome to the NumPy Course for Viral Clip Generation!")
        print("This course will teach you everything you need to know about NumPy.")
        print("Each lesson builds on the previous one, so follow along carefully.")
        
        lessons = [
            self.lesson_1_basics,
            self.lesson_2_indexing,
            self.lesson_3_operations,
            self.lesson_4_reshaping,
            self.lesson_5_linear_algebra,
            self.lesson_6_random_numbers,
            self.lesson_7_performance,
            self.lesson_8_practical_applications
        ]
        
        for lesson in lessons:
            lesson()
            self.current_lesson += 1
            
        self.final_assessment()
        
        print("\n" + "="*60)
        print("🎉 CONGRATULATIONS! You've completed the NumPy course!")
        print("="*60)
        print("\nWhat you've learned:")
        print("✅ Creating and manipulating arrays")
        print("✅ Indexing and slicing")
        print("✅ Array operations and broadcasting")
        print("✅ Linear algebra operations")
        print("✅ Random number generation")
        print("✅ Performance optimization")
        print("✅ Practical applications for viral clip generation")
        
        print("\nNext steps:")
        print("1. Practice with the exercises in each lesson")
        print("2. Apply NumPy to your own projects")
        print("3. Move on to the next library (Pandas, Matplotlib, etc.)")
        print("4. Build your viral clip generation pipeline!")
        
        print(f"\nYour final score: {self.score}/{self.max_score}")

def main():
    """Main function to run the course"""
    course = NumPyCourse()
    course.run_course()

if __name__ == "__main__":
    main()
