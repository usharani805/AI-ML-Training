import numpy as np
import pytest

from day2_numpy_basics import (
    create_identity_matrix,
    random_matrix_stats,
    reshape_pipeline,
    stack_arrays,
)


def test_identity_matrix_n1():
    result = create_identity_matrix(1)

    assert np.array_equal(result, np.array([[1.0]]))


def test_identity_matrix_n5():
    result = create_identity_matrix(5)

    assert np.array_equal(result, np.eye(5))


def test_random_matrix_stats():
    result = random_matrix_stats(2, 3)

    assert set(result.keys()) == {"min", "max", "mean", "std"}
    assert result["min"] <= result["max"]


def test_reshape_valid():
    arr = np.array([1, 2, 3, 4])

    result = reshape_pipeline(arr, (2, 2))

    assert np.array_equal(
        result,
        np.array([[1, 2], [3, 4]])
    )


def test_reshape_invalid():
    arr = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        reshape_pipeline(arr, (3, 3))


def test_stack_arrays():
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])

    result = stack_arrays([arr1, arr2], axis=0)

    assert np.array_equal(
        result,
        np.array([[1, 2], [3, 4]])
    )