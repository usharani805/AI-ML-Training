import numpy as np
# Basic slicing - 1D array
arr_1d = np.array([10, 20, 30, 40, 50])

print("1D array:", arr_1d)
print("1D slice:", arr_1d[1:4])


# Basic slicing - 2D array
arr_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("2D array:")
print(arr_2d)

print("Rows:", arr_2d[0:2, :])
print("Columns:", arr_2d[:, 1:3])
print("Sub-matrix:", arr_2d[0:2, 1:3])
# Boolean masking
arr = np.array([10, 25, 30, 45, 50, 65])

mask = arr > 30

print("Boolean mask:", mask)
print("Filtered elements:", arr[mask])
# Fancy indexing
arr_fancy = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [100, 110, 120]
])

row_indices = np.array([0, 2, 3])

selected_rows = arr_fancy[row_indices]

print("Selected rows using fancy indexing:")
print(selected_rows)
# Broadcasting example 1: Scalar + Array

arr_broadcast = np.array([10, 20, 30, 40])

result = arr_broadcast + 5

print("Scalar + Array:")
print(result)
# Broadcasting example 2: 1D + 2D

arr_2d_broadcast = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

arr_1d = np.array([1, 2, 3])

result = arr_2d_broadcast + arr_1d

print("1D + 2D:")
print(result)
# Broadcasting example 3: Mismatched shapes

arr_a = np.array([1, 2, 3])
arr_b = np.array([10, 20])

try:
    result = arr_a + arr_b
    print("Result:", result)
except ValueError as e:
    print("Broadcasting error:", e)
    # Broadcasting Rules:
# 1. Shapes are compared from right to left.
# 2. Two dimensions are compatible when they are equal,
#    or when one of them is 1.
#
# Worked Example 1:
# (4,) + scalar -> compatible
# Example: [10, 20, 30, 40] + 5
#
# Worked Example 2:
# (2, 3) + (3,) -> compatible
# Example: [[10,20,30],[40,50,60]] + [1,2,3]
#
# Worked Example 3:
# (3,) + (2,) -> not compatible
# Example: [1,2,3] + [10,20] -> ValueError
def filter_outliers(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return elements whose absolute value is within the threshold."""
    mask = np.abs(arr) <= threshold
    return arr[mask]
def extract_submatrix(
    arr: np.ndarray,
    row_range: tuple,
    col_range: tuple
) -> np.ndarray:
    """Extract a sub-matrix using row and column ranges."""
    return arr[row_range[0]:row_range[1], col_range[0]:col_range[1]]
def normalize_broadcast(arr: np.ndarray) -> np.ndarray:
    """Normalize each column using broadcasting."""
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    if np.any(std == 0):
        raise ValueError("Cannot normalize columns with zero standard deviation")

    return (arr - mean) / std
def select_rows_by_index(
    arr: np.ndarray,
    indices: list
) -> np.ndarray:
    """Select specific rows using fancy indexing."""
    return arr[np.array(indices)]
# normalize_broadcast demonstration

sample_data = np.array([
    [10, 100, 1000],
    [20, 200, 2000],
    [30, 300, 3000],
    [40, 400, 4000]
])

print("Before normalization:")
print(sample_data)

normalized_data = normalize_broadcast(sample_data)

print("After normalization:")
print(normalized_data)

print("Column means:", np.mean(normalized_data, axis=0))
print("Column standard deviations:", np.std(normalized_data, axis=0))
# ============================================================
# 9. Outlier Filtering Test
# ============================================================

outlier_data = np.array([-2, -1, 0, 1, 2, 10])

filtered_data = filter_outliers(
    outlier_data,
    threshold=2
)

print("Original data:", outlier_data)
print("Filtered data:", filtered_data)

