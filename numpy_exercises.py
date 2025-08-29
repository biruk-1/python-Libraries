#!/usr/bin/env python3
"""
NumPy Practice Exercises
Additional exercises to reinforce your NumPy learning
"""

import numpy as np

def exercise_1_basic_arrays():
    """Exercise 1: Basic Array Creation"""
    print("="*50)
    print("EXERCISE 1: Basic Array Creation")
    print("="*50)
    
    # Your tasks:
    print("Create the following arrays:")
    print("1. A 1D array with numbers 1 to 10")
    print("2. A 2D array (3x3) filled with zeros")
    print("3. A 2D array (4x4) filled with ones")
    print("4. A 2D array (3x3) with random numbers between 0 and 1")
    print("5. A 1D array with 5 evenly spaced numbers from 0 to 1")
    
    # Solutions (uncomment to see):
    print("\n--- Solutions ---")
    print("1. np.arange(1, 11) or np.array([1,2,3,4,5,6,7,8,9,10])")
    print("2. np.zeros((3, 3))")
    print("3. np.ones((4, 4))")
    print("4. np.random.rand(3, 3)")
    print("5. np.linspace(0, 1, 5)")
    
    # Try them:
    print("\n--- Try them yourself ---")
    arr1 = np.arange(1, 11)
    arr2 = np.zeros((3, 3))
    arr3 = np.ones((4, 4))
    arr4 = np.random.rand(3, 3)
    arr5 = np.linspace(0, 1, 5)
    
    print(f"1. {arr1}")
    print(f"2. \n{arr2}")
    print(f"3. \n{arr3}")
    print(f"4. \n{arr4}")
    print(f"5. {arr5}")

def exercise_2_indexing():
    """Exercise 2: Array Indexing and Slicing"""
    print("\n" + "="*50)
    print("EXERCISE 2: Array Indexing and Slicing")
    print("="*50)
    
    # Create a sample array
    arr = np.array([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]])
    
    print(f"Sample array:\n{arr}")
    print(f"Shape: {arr.shape}")
    
    print("\nExtract the following elements:")
    print("1. The element at row 1, column 2")
    print("2. The entire second row")
    print("3. The entire third column")
    print("4. A 2x2 subarray starting from row 1, column 1")
    print("5. All elements greater than 10")
    
    print("\n--- Solutions ---")
    print("1. arr[1, 2]")
    print("2. arr[1, :] or arr[1]")
    print("3. arr[:, 2]")
    print("4. arr[1:3, 1:3]")
    print("5. arr[arr > 10]")
    
    print("\n--- Results ---")
    print(f"1. {arr[1, 2]}")
    print(f"2. {arr[1, :]}")
    print(f"3. {arr[:, 2]}")
    print(f"4. \n{arr[1:3, 1:3]}")
    print(f"5. {arr[arr > 10]}")

def exercise_3_operations():
    """Exercise 3: Array Operations"""
    print("\n" + "="*50)
    print("EXERCISE 3: Array Operations")
    print("="*50)
    
    # Create sample arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    print("\nPerform these operations:")
    print("1. Add arrays a and b")
    print("2. Multiply array a by 2")
    print("3. Calculate the square root of array a")
    print("4. Find the mean of array b")
    print("5. Find the maximum value in array a")
    
    print("\n--- Solutions ---")
    print("1. a + b")
    print("2. a * 2")
    print("3. np.sqrt(a)")
    print("4. np.mean(b) or b.mean()")
    print("5. np.max(a) or a.max()")
    
    print("\n--- Results ---")
    print(f"1. {a + b}")
    print(f"2. {a * 2}")
    print(f"3. {np.sqrt(a)}")
    print(f"4. {np.mean(b)}")
    print(f"5. {np.max(a)}")

def exercise_4_reshaping():
    """Exercise 4: Reshaping Arrays"""
    print("\n" + "="*50)
    print("EXERCISE 4: Reshaping Arrays")
    print("="*50)
    
    # Create a 1D array
    arr = np.arange(12)
    print(f"Original array: {arr}")
    print(f"Shape: {arr.shape}")
    
    print("\nReshape the array as follows:")
    print("1. Into a 3x4 matrix")
    print("2. Into a 4x3 matrix")
    print("3. Into a 2x2x3 3D array")
    print("4. Transpose the 3x4 matrix")
    print("5. Flatten the 3x4 matrix back to 1D")
    
    print("\n--- Solutions ---")
    print("1. arr.reshape(3, 4)")
    print("2. arr.reshape(4, 3)")
    print("3. arr.reshape(2, 2, 3)")
    print("4. arr.reshape(3, 4).T")
    print("5. arr.reshape(3, 4).flatten()")
    
    print("\n--- Results ---")
    arr_3x4 = arr.reshape(3, 4)
    print(f"1. \n{arr_3x4}")
    print(f"2. \n{arr.reshape(4, 3)}")
    print(f"3. \n{arr.reshape(2, 2, 3)}")
    print(f"4. \n{arr_3x4.T}")
    print(f"5. {arr_3x4.flatten()}")

def exercise_5_linear_algebra():
    """Exercise 5: Linear Algebra"""
    print("\n" + "="*50)
    print("EXERCISE 5: Linear Algebra")
    print("="*50)
    
    # Create matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    
    print("\nPerform these operations:")
    print("1. Matrix multiplication of A and B")
    print("2. Element-wise multiplication of A and B")
    print("3. Calculate the determinant of A")
    print("4. Calculate the inverse of A")
    print("5. Solve the equation Ax = [5, 11] for x")
    
    print("\n--- Solutions ---")
    print("1. A @ B or np.dot(A, B)")
    print("2. A * B")
    print("3. np.linalg.det(A)")
    print("4. np.linalg.inv(A)")
    print("5. np.linalg.solve(A, [5, 11])")
    
    print("\n--- Results ---")
    print(f"1. \n{A @ B}")
    print(f"2. \n{A * B}")
    print(f"3. {np.linalg.det(A)}")
    print(f"4. \n{np.linalg.inv(A)}")
    b = np.array([5, 11])
    x = np.linalg.solve(A, b)
    print(f"5. x = {x}")
    print(f"   Verification: A @ x = {A @ x}")

def exercise_6_practical_applications():
    """Exercise 6: Practical Applications"""
    print("\n" + "="*50)
    print("EXERCISE 6: Practical Applications")
    print("="*50)
    
    print("Let's apply NumPy to viral clip generation scenarios:")
    
    # 1. Image processing
    print("\n1. Image Processing:")
    # Create a simple image (grayscale)
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    print(f"   Created image shape: {image.shape}")
    print(f"   Image statistics: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
    
    # Apply a simple filter (brightness adjustment)
    brightened = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    print(f"   Brightened image mean: {brightened.mean():.1f}")
    
    # 2. Audio processing
    print("\n2. Audio Processing:")
    # Create a simple audio signal
    sample_rate = 22050
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A note
    audio = np.sin(2 * np.pi * frequency * t)
    
    print(f"   Audio signal shape: {audio.shape}")
    print(f"   Signal statistics: min={audio.min():.3f}, max={audio.max():.3f}, mean={audio.mean():.3f}")
    
    # Add some noise
    noise = np.random.normal(0, 0.1, audio.shape)
    noisy_audio = audio + noise
    print(f"   Noisy signal mean: {noisy_audio.mean():.3f}")
    
    # 3. Feature extraction
    print("\n3. Feature Extraction:")
    # Simulate video features
    num_videos = 50
    features = np.random.randn(num_videos, 5)  # 5 features per video
    
    # Calculate basic statistics
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    
    print(f"   Number of videos: {num_videos}")
    print(f"   Features per video: {features.shape[1]}")
    print(f"   Feature means: {feature_means}")
    print(f"   Feature standard deviations: {feature_stds}")
    
    # 4. Viral score calculation
    print("\n4. Viral Score Calculation:")
    # Define feature weights
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Calculate viral scores
    viral_scores = np.dot(features, weights)
    
    print(f"   Viral scores range: [{viral_scores.min():.3f}, {viral_scores.max():.3f}]")
    print(f"   Average viral score: {viral_scores.mean():.3f}")
    
    # Find top videos
    top_5_indices = np.argsort(viral_scores)[-5:]
    print(f"   Top 5 viral video indices: {top_5_indices}")
    print(f"   Top 5 viral scores: {viral_scores[top_5_indices]}")

def run_all_exercises():
    """Run all exercises"""
    print("🧪 NumPy Practice Exercises")
    print("Complete these exercises to reinforce your learning!")
    
    exercises = [
        exercise_1_basic_arrays,
        exercise_2_indexing,
        exercise_3_operations,
        exercise_4_reshaping,
        exercise_5_linear_algebra,
        exercise_6_practical_applications
    ]
    
    for exercise in exercises:
        exercise()
        input("\nPress Enter to continue to the next exercise...")
    
    print("\n" + "="*60)
    print("🎉 All exercises completed!")
    print("="*60)
    print("\nYou've practiced:")
    print("✅ Basic array creation and manipulation")
    print("✅ Indexing and slicing techniques")
    print("✅ Array operations and mathematical functions")
    print("✅ Reshaping and transposing arrays")
    print("✅ Linear algebra operations")
    print("✅ Practical applications for viral clip generation")
    
    print("\nNext steps:")
    print("1. Try modifying the exercises with your own data")
    print("2. Experiment with different array shapes and operations")
    print("3. Apply these concepts to your viral clip generation project")
    print("4. Move on to learning the next library!")

if __name__ == "__main__":
    run_all_exercises()
