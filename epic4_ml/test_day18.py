import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from epic4_ml.day18_linear_regression import (
    gradient_descent_fit,
    normal_equation_fit,
    regression_metrics,
)


def test_normal_equation_coefficients():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 3 * X.ravel() + 2

    coefficients = normal_equation_fit(X, y)

    assert np.allclose(coefficients, [2, 3], atol=1e-6)


def test_normal_equation_matches_sklearn():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([5.1, 7.9, 11.2, 13.8, 17.1])

    scratch_coefficients = normal_equation_fit(X, y)

    model = LinearRegression()
    model.fit(X, y)

    sklearn_coefficients = np.concatenate(
        [[model.intercept_], model.coef_]
    )

    assert np.allclose(
        scratch_coefficients,
        sklearn_coefficients,
        atol=1e-6,
    )


def test_gradient_descent_loss_decreases():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 3 * X.ravel() + 2

    _, loss_history = gradient_descent_fit(
        X,
        y,
        lr=0.01,
        epochs=100,
    )

    assert loss_history[-1] < loss_history[0]


def test_gradient_descent_loss_monotonic():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 3 * X.ravel() + 2

    _, loss_history = gradient_descent_fit(
        X,
        y,
        lr=0.01,
        epochs=100,
    )

    assert all(
        loss_history[i + 1] <= loss_history[i] + 1e-12
        for i in range(len(loss_history) - 1)
    )


def test_regression_mae():
    y_true = np.array([1, 2, 3, 4], dtype=float)
    y_pred = np.array([1.1, 1.9, 3.2, 3.8], dtype=float)

    metrics = regression_metrics(y_true, y_pred)

    expected = mean_absolute_error(y_true, y_pred)

    assert np.isclose(metrics["MAE"], expected, atol=1e-6)


def test_regression_mse_rmse():
    y_true = np.array([1, 2, 3, 4], dtype=float)
    y_pred = np.array([1.1, 1.9, 3.2, 3.8], dtype=float)

    metrics = regression_metrics(y_true, y_pred)

    expected_mse = mean_squared_error(y_true, y_pred)
    expected_rmse = np.sqrt(expected_mse)

    assert np.isclose(metrics["MSE"], expected_mse, atol=1e-6)
    assert np.isclose(metrics["RMSE"], expected_rmse, atol=1e-6)


def test_regression_r2():
    y_true = np.array([1, 2, 3, 4], dtype=float)
    y_pred = np.array([1.1, 1.9, 3.2, 3.8], dtype=float)

    metrics = regression_metrics(y_true, y_pred)

    expected = r2_score(y_true, y_pred)

    assert np.isclose(metrics["R2"], expected, atol=1e-6)


def test_singular_matrix_handled():
    X = np.array(
        [
            [1, 2],
            [2, 4],
            [3, 6],
            [4, 8],
        ],
        dtype=float,
    )

    y = np.array([3, 6, 9, 12], dtype=float)

    coefficients = normal_equation_fit(X, y)

    assert np.all(np.isfinite(coefficients))