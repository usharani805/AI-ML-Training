import numpy as np


def create_identity_matrix(n: int) -> np.ndarray:
    return np.eye(n)


def random_matrix_stats(rows: int, cols: int) -> dict:
    matrix = np.random.rand(rows, cols)

    return {
        "min": np.min(matrix),
        "max": np.max(matrix),
        "mean": np.mean(matrix),
        "std": np.std(matrix),
    }


def reshape_pipeline(arr: np.ndarray, shape: tuple) -> np.ndarray:
    try:
        return arr.reshape(shape)
    except ValueError as error:
        raise ValueError("Invalid reshape shape") from error


def stack_arrays(
    arr_list: list[np.ndarray], axis: int
) -> np.ndarray:
    return np.stack(arr_list, axis=axis)


def inspect_array(arr: np.ndarray) -> None:
    print("\nArray Inspector")
    print("----------------")
    print(f"Shape       : {arr.shape}")
    print(f"Dtype       : {arr.dtype}")
    print(f"Memory Size : {arr.nbytes} bytes")
    print(f"Dimensions  : {arr.ndim}")


# Array creation from a list
numbers = np.array([1, 2, 3, 4, 5])
print("Array from list:", numbers)

# Create array with zeros
zeros_array = np.zeros(5)
print("Zeros array:", zeros_array)

# Create array with ones
ones_array = np.ones(5)
print("Ones array:", ones_array)

# Create array using arange
range_array = np.arange(1, 6)
print("Arange array:", range_array)

# Create array using linspace
linear_array = np.linspace(0, 1, 5)
print("Linspace array:", linear_array)

# Create identity matrix
identity_array = np.eye(3)
print("Identity matrix:")
print(identity_array)

# Create random array
random_array = np.random.rand(2, 3)
print("Random array:")
print(random_array)

# Array properties
print("Shape:", random_array.shape)
print("Dimensions:", random_array.ndim)
print("Data type:", random_array.dtype)
print("Size:", random_array.size)
print("Item size:", random_array.itemsize)

# 1D, 2D and 3D arrays
array_1d = np.array([1, 2, 3, 4])
array_2d = np.array([[1, 2], [3, 4]])
array_3d = np.array([[[1, 2], [3, 4]]])

print("\n1D Array:")
print("Shape:", array_1d.shape)
print("Dimensions:", array_1d.ndim)
print("Dtype:", array_1d.dtype)
print("Size:", array_1d.size)
print("Item size:", array_1d.itemsize)

print("\n2D Array:")
print("Shape:", array_2d.shape)
print("Dimensions:", array_2d.ndim)
print("Dtype:", array_2d.dtype)
print("Size:", array_2d.size)
print("Item size:", array_2d.itemsize)

print("\n3D Array:")
print("Shape:", array_3d.shape)
print("Dimensions:", array_3d.ndim)
print("Dtype:", array_3d.dtype)
print("Size:", array_3d.size)
print("Item size:", array_3d.itemsize)

# Reshape
array = np.array([1, 2, 3, 4, 5, 6])
reshaped_array = array.reshape(2, 3)

print("\nReshaped array:")
print(reshaped_array)

# flatten() creates a new copy.
flattened_array = reshaped_array.flatten()

# ravel() returns a flattened view when possible.
ravel_array = reshaped_array.ravel()

print("\nFlattened array:")
print(flattened_array)

print("\nRavel array:")
print(ravel_array)

# Concatenation
array_a = np.array([1, 2, 3])
array_b = np.array([4, 5, 6])

concatenated = np.concatenate((array_a, array_b))
vertical = np.vstack((array_a, array_b))
horizontal = np.hstack((array_a, array_b))

print("\nConcatenate:")
print(concatenated)

print("\nVertical stack:")
print(vertical)

print("\nHorizontal stack:")
print(horizontal)

# Practical functions
print("\nIdentity Matrix Function:")
print(create_identity_matrix(3))

print("\nRandom Matrix Statistics:")
print(random_matrix_stats(3, 3))

print("\nReshape Pipeline:")
print(reshape_pipeline(np.array([1, 2, 3, 4]), (2, 2)))

print("\nStack Arrays:")
print(stack_arrays([np.array([1, 2]), np.array([3, 4])], axis=0))

# Array Inspector
print("\nInspector for 1D array:")
inspect_array(array_1d)

print("\nInspector for 2D array:")
inspect_array(array_2d)

print("\nInspector for 3D array:")
inspect_array(array_3d)

# Dtype casting
int_array = np.array([1, 2, 3, 4], dtype=np.int32)
float_array = int_array.astype(np.float64)

print("\nDtype Casting:")
print("Original dtype:", int_array.dtype)
print("New dtype:", float_array.dtype)
print("Original memory:", int_array.nbytes, "bytes")
print("New memory:", float_array.nbytes, "bytes")

# float64 uses more memory than int32 for each element.

# Memory usage comparison
print("\nMemory Usage Comparison:")

for dtype in [np.int32, np.int64, np.float32, np.float64]:
    arr = np.zeros((1000, 1000), dtype=dtype)
    print(
        f"{dtype.__name__:>8} : "
        f"{arr.nbytes / (1024 * 1024):.2f} MB"
    )