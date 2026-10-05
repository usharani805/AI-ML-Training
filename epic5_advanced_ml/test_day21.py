import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from epic5_advanced_ml.day21_knn_svm import (
    compare_kernels,
    euclidean_distance_matrix,
    k_sweep_analysis,
    knn_predict_from_scratch,
    plot_svm_decision_boundary,
)


def test_euclidean_distance_matrix():
    X1 = np.array([[0, 0], [3, 4]])
    X2 = np.array([[0, 0]])

    distances = euclidean_distance_matrix(X1, X2)

    assert distances.shape == (2, 1)
    assert distances[0, 0] == 0
    assert distances[1, 0] == 5


def test_scratch_knn_matches_sklearn():
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [5, 5],
            [5, 6],
            [6, 5],
        ]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[0.2, 0.2], [5.2, 5.2]])

    scratch_predictions = knn_predict_from_scratch(
        X_train, y_train, X_test, k=3
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    sklearn_predictions = model.predict(X_test)

    assert np.array_equal(scratch_predictions, sklearn_predictions)


def test_invalid_k_raises_value_error():
    X_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])
    X_test = np.array([[0, 0]])

    with pytest.raises(ValueError):
        knn_predict_from_scratch(
            X_train, y_train, X_test, k=0
        )


def test_k_greater_than_training_samples_raises_value_error():
    X_train = np.array([[0, 0], [1, 1]])
    y_train = np.array([0, 1])
    X_test = np.array([[0, 0]])

    with pytest.raises(ValueError):
        knn_predict_from_scratch(
            X_train, y_train, X_test, k=3
        )


def test_k_sweep_returns_one_row_per_k():
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [5, 5],
            [5, 6],
            [6, 5],
        ]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[0.2, 0.2], [5.2, 5.2]])
    y_test = np.array([0, 1])

    k_range = range(1, 4)

    result = k_sweep_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        k_range,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result["k"]) == [1, 2, 3]
    assert "accuracy" in result.columns


def test_compare_kernels_returns_expected_columns():
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [5, 5],
            [5, 6],
            [6, 5],
        ]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[0.2, 0.2], [5.2, 5.2]])
    y_test = np.array([0, 1])

    result = compare_kernels(
        X_train,
        X_test,
        y_train,
        y_test,
        ["linear", "rbf", "poly"],
    )

    assert len(result) == 3
    assert set(result["kernel"]) == {
        "linear",
        "rbf",
        "poly",
    }
    assert "accuracy" in result.columns
    assert "f1" in result.columns
    assert "training_time" in result.columns


def test_plot_svm_decision_boundary_creates_file(tmp_path):
    from sklearn.svm import SVC

    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [5, 5],
            [5, 6],
            [6, 5],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = SVC(kernel="linear")
    model.fit(X, y)

    output_file = tmp_path / "svm_boundary.png"

    plot_svm_decision_boundary(
        model,
        X,
        y,
        str(output_file),
    )

    assert output_file.exists()