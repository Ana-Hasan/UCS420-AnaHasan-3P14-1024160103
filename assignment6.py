
import numpy as np

# ============================================================
# Q1. SENSOR READINGS: NUMPY VECTORIZATION AND BOOLEAN INDEXING
# ============================================================

print("=" * 60)
print("Q1: SENSOR READINGS")
print("=" * 60)

# Original temperature readings
temperature = np.array([28, 31, 35, 38, 27, 33, 40])

# a. Add 2°C to every reading using vectorization
corrected_temperature = temperature + 2

print("Original temperatures:", temperature)
print("Corrected temperatures (+2°C):", corrected_temperature)

# b. Convert Celsius to Fahrenheit using vectorization
fahrenheit = (9 / 5) * corrected_temperature + 32

print("Temperatures in Fahrenheit:", fahrenheit)

# c. Identify readings greater than 32°C using Boolean indexing
greater_than_32 = corrected_temperature > 32
readings_above_32 = corrected_temperature[greater_than_32]

print("Boolean mask:", greater_than_32)
print("Readings greater than 32°C:", readings_above_32)

# d. Count readings exceeding 32°C
count_above_32 = np.sum(greater_than_32)

print("Number of readings exceeding 32°C:", count_above_32)

# e. Explain efficiency
print("""
Explanation:
Vectorization applies an operation to the entire NumPy array
without explicitly writing a Python loop. Boolean indexing
selects elements using a condition. These operations are
generally faster and more concise than explicit Python loops.
""")


# ============================================================
# Q2. DAILY STEP COUNTS: NUMPY ARRAY OPERATIONS
# ============================================================

print("\n" + "=" * 60)
print("Q2: DAILY STEP COUNTS")
print("=" * 60)

# Rows represent users; columns represent days
steps = np.array([
    [5000, 6200, 7100],
    [8000, 7500, 9000],
    [4500, 5100, 4800],
    [9000, 8500, 9500]
])

print("Steps array:\n", steps)

# a. Calculate total steps recorded
total_steps = np.sum(steps)

print("Total steps recorded:", total_steps)

# b. Calculate mean number of steps
mean_steps = np.mean(steps)

print("Mean steps:", mean_steps)

# c. Find maximum and minimum values
maximum_steps = np.max(steps)
minimum_steps = np.min(steps)

print("Maximum steps:", maximum_steps)
print("Minimum steps:", minimum_steps)

# d. Calculate total steps for each day using axis=0
daily_totals = np.sum(steps, axis=0)

print("Total steps for each day (axis=0):", daily_totals)

# e. Calculate total steps for each user using axis=1
user_totals = np.sum(steps, axis=1)

print("Total steps for each user (axis=1):", user_totals)

# f. Find the user with the maximum number of steps
# using row-wise totals
most_active_user = np.argmax(user_totals) + 1

print("Most active user: User", most_active_user)
print("Their total steps:", np.max(user_totals))


# ============================================================
# Q3. NUMPY ARRAY CREATION, COPYING, SLICING AND RESHAPING
# ============================================================

print("\n" + "=" * 60)
print("Q3: NUMPY ARRAY OPERATIONS")
print("=" * 60)

# a. Create the original NumPy array
original = np.array([1, 2, 3, 4, 5, 6])

print("Original array:", original)

# b. Create a slice containing elements from index 1 to 4
subset = original[1:5]

print("Subset (index 1 to 4):", subset)

# c. Modify the first element of subset to 999
subset[0] = 999

print("Original after modifying subset:", original)
print("Modified subset:", subset)

print("""
Explanation:
A slice is generally a view of the original array.
Therefore, modifying subset also changes original.
""")

# d. Create another slice using copy()
original = np.array([1, 2, 3, 4, 5, 6])
subset_copy = original[1:5].copy()

subset_copy[0] = 500

print("Original after modifying copied subset:", original)
print("Copied and modified subset:", subset_copy)

print("""
Explanation:
copy() creates an independent array. Changes to the copy
do not affect the original array.
""")

# e. Create a NumPy array from 1 to 12 and reshape to 3x4
array_3x4 = np.arange(1, 13).reshape(3, 4)

print("3x4 array:\n", array_3x4)

# f. Extract elements using indexing and slicing

# First row
print("First row:", array_3x4[0, :])

# Last row
print("Last row:", array_3x4[-1, :])

# Second column
print("Second column:", array_3x4[:, 1])

# Elements from rows 1-2 and columns 2-3
# Using zero-based indexing: rows 1,2 and columns 2,3
print("Rows 1-2, columns 2-3:\n", array_3x4[1:3, 2:4])

# g. Create a flattened version using flatten() and ravel()
flattened = array_3x4.flatten()
raveled = array_3x4.ravel()

print("Flattened array:", flattened)
print("Raveled array:", raveled)

# h. Modify an element in the array returned by ravel()
raveled[0] = 100

print("After modifying ravel result:")
print("Raveled array:", raveled)
print("Original 3x4 array:\n", array_3x4)

print("""
Explanation:
For this contiguous array, ravel() usually returns a view,
so modifying it also changes the original array.
""")

# i. Modify an element in the array returned by flatten()
flattened[1] = 200

print("After modifying flatten result:")
print("Flattened array:", flattened)
print("Original 3x4 array:\n", array_3x4)

print("""
Explanation:
flatten() always returns a copy. Therefore, modifying the
flattened array does not affect the original array.
""")

# j. Display shape, ndim, size and dtype of the 3x4 array
print("Shape:", array_3x4.shape)
print("Number of dimensions:", array_3x4.ndim)
print("Total number of elements:", array_3x4.size)
print("Data type:", array_3x4.dtype)


# ============================================================
# Q4. COGNITIVE ASSISTANCE SCORE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("Q4: COGNITIVE ASSISTANCE SCORE PREDICTION")
print("=" * 60)

# Each row represents a user
# Columns: Sleep hours, Activity level, Stress level
X = np.array([
    [6, 70, 3],
    [5, 60, 5],
    [8, 80, 2],
    [4, 30, 8]
])

# Assistance scores
y = np.array([40, 65, 30, 85])

print("Feature matrix X:\n", X)
print("Target vector y:", y)

# a. Display shape and dimensions of X
print("Shape of X:", X.shape)
print("Number of dimensions of X:", X.ndim)

# b. Calculate X.T and explain transpose
X_transpose = X.T

print("Transpose of X:\n", X_transpose)

print("""
Explanation:
Transpose swaps rows and columns. X has shape (4, 3),
while X.T has shape (3, 4).
""")

# c. Calculate the matrix product X.T @ X
product = X.T @ X

print("X.T @ X:\n", product)

# d. Calculate the matrix inverse using NumPy
# X.T @ X is not invertible for these given data because
# the three feature columns are linearly dependent.
# Use the pseudoinverse to handle this safely.

inverse_matrix = np.linalg.pinv(product)

print("Pseudoinverse of X.T @ X:\n", inverse_matrix)

# e. Ordinary Least Squares:
# beta = (X.T @ X)^(-1) @ X.T @ y
# Use the pseudoinverse for numerical stability.

beta = np.linalg.pinv(X.T @ X) @ X.T @ y

print("Regression coefficients (beta):", beta)

# Predicted scores for the training data
predicted_scores = X @ beta

print("Predicted scores:", predicted_scores)

# f. Explain the resulting coefficients
print("""
Explanation:
The coefficients represent the contribution of each feature
to the predicted score in this linear model. Since the
features are correlated and the matrix is rank-deficient,
the individual coefficients are not uniquely identifiable.
The pseudoinverse returns a minimum-norm least-squares
solution.
""")

# g. Predict the assistance score for a new user
# Sleep = 5 hours, Activity = 40, Stress = 7
new_user = np.array([5, 40, 7])

predicted_assistance_score = new_user @ beta

print("New user features:", new_user)
print("Predicted assistance score:", predicted_assistance_score)

print("\n" + "=" * 60)
print("ALL QUESTIONS COMPLETED SUCCESSFULLY")
print("=" * 60)
