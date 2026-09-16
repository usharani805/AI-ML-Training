import numpy as np
import pytest

from day3_indexing_broadcasting import (
    filter_outliers,
    extract_submatrix,
    normalize_broadcast,
    select_rows_by_index
)


# ============================================================
# 1. Indexing Test
# ============================================================

def test_basic_indexing():
    arr = np.array([10, 20, 30, 40, 50])

    assert arr[0] == 10
    assert arr[-1] == 50


# ============================================================
# 2. Slicing Test
# ============================================================

def test_slicing():
    arr = np.array([10, 20, 30, 40, 50])

    result = arr[1:4]

    np.testing.assert_array_equal(
        result,
        np.array([20, 30, 40])
    )


# ============================================================
# 3. Boolean Masking Test
# ============================================================

def test_boolean_masking():
    arr = np.array([10, 20, 30, 40, 50])

    result = arr[arr > 25]

    np.testing.assert_array_equal(
        result,
        np.array([30, 40, 50])
    )


# ============================================================
# 4. Extract Submatrix Test
# ============================================================

def test_extract_submatrix():
    arr = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    result = extract_submatrix(
        arr,
        (0, 2),
        (1, 3)
    )

    expected = np.array([
        [2, 3],
        [5, 6]
    ])

    np.testing.assert_array_equal(result, expected)


# ============================================================
# 5. Fancy Indexing Test
# ============================================================

def test_fancy_indexing():
    arr = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])

    result = select_rows_by_index(
        arr,
        [0, 2]
    )

    expected = np.array([
        [10, 20, 30],
        [70, 80, 90]
    ])

    np.testing.assert_array_equal(result, expected)


# ============================================================
# 6. Normalization Mean Test
# ============================================================

def test_normalize_mean():
    arr = np.array([
        [10, 20],
        [20, 30],
        [30, 40],
        [40, 50]
    ])

    result = normalize_broadcast(arr)

    means = np.mean(result, axis=0)

    np.testing.assert_allclose(
        means,
        np.zeros(2),
        atol=1e-7
    )


# ============================================================
# 7. Normalization Standard Deviation Test
# ============================================================

def test_normalize_std():
    arr = np.array([
        [10, 20],
        [20, 30],
        [30, 40],
        [40, 50]
    ])

    result = normalize_broadcast(arr)

    stds = np.std(result, axis=0)

    np.testing.assert_allclose(
        stds,
        np.ones(2),
        atol=1e-7
    )


# ============================================================
# 8. Outlier Filtering - Some Outliers
# ============================================================

def test_filter_outliers():
    arr = np.array([-2, -1, 0, 1, 2, 10])

    result = filter_outliers(
        arr,
        threshold=2
    )

    expected = np.array([-2, -1, 0, 1, 2])

    np.testing.assert_array_equal(result, expected)


# ============================================================
# 9. Outlier Filtering - No Outliers
# ============================================================

def test_filter_outliers_no_outliers():
    arr = np.array([-2, -1, 0, 1, 2])

    result = filter_outliers(
        arr,
        threshold=2
    )

    np.testing.assert_array_equal(result, arr)


# ============================================================
# 10. Outlier Filtering - All Outliers
# ============================================================

def test_filter_outliers_all_outliers():
    arr = np.array([10, 20, 30])

    result = filter_outliers(
        arr,
        threshold=2
    )

    assert result.size == 0


# ============================================================
# 11. Broadcasting Failure Test
# ============================================================

def test_broadcasting_failure():
    a = np.array([1, 2, 3])
    b = np.array([10, 20])

    with pytest.raises(ValueError):
        a + b