from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from epic4_ml.day16_ml_basics import (
    california_X_train,
    california_X_test,
    california_y_train,
    california_y_test,
)
from epic4_ml.day17_feature_engineering import add_engineered_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.csv"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
CHARTS_DIR = BASE_DIR / "charts"


def normal_equation_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit linear regression using the normal equation."""

    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

    try:
        coefficients = (
            np.linalg.inv(X_with_bias.T @ X_with_bias)
            @ X_with_bias.T
            @ y
        )
    except np.linalg.LinAlgError:
        coefficients = (
            np.linalg.pinv(X_with_bias.T @ X_with_bias)
            @ X_with_bias.T
            @ y
        )

    return coefficients


def gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, list]:
    """Fit linear regression using gradient descent."""

    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

    weights = np.zeros(X_with_bias.shape[1])
    loss_history = []

    for _ in range(epochs):
        predictions = X_with_bias @ weights
        errors = predictions - y

        loss = np.mean(errors**2)
        loss_history.append(loss)

        gradients = (2 / len(y)) * (X_with_bias.T @ errors)
        weights -= lr * gradients

    return weights, loss_history


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate regression metrics from scratch."""

    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Save residual plot and predicted-vs-actual plot."""

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")

    axes[1].scatter(y_true, y_pred, alpha=0.6)
    axes[1].set_xlabel("Actual Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].set_title("Predicted vs Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Residuals should be reasonably scattered around zero.
    # A clear pattern may indicate that the model misses some relationship.


if __name__ == "__main__":
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    target = "MedHouseVal"

    X = df.drop(columns=[target])
    y = df[target]

    X = add_engineered_features(X)

    # Use the same train/test split created in Day16.
    X_train_original = X.loc[california_X_train.index]
    X_test_original = X.loc[california_X_test.index]

    y_train = y.loc[california_y_train.index].to_numpy()
    y_test = y.loc[california_y_test.index].to_numpy()

    # Use the preprocessor created and saved in Day17.
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_train = preprocessor.transform(X_train_original)
    X_test = preprocessor.transform(X_test_original)

    # From-scratch normal equation.
    scratch_coefficients = normal_equation_fit(
        X_train,
        y_train,
    )

    scratch_predictions = np.column_stack(
        [np.ones(X_test.shape[0]), X_test]
    ) @ scratch_coefficients

    # From-scratch gradient descent.
    gradient_weights, loss_history = gradient_descent_fit(
        X_train,
        y_train,
        lr=0.01,
        epochs=1000,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Gradient Descent Loss Curve")
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "linear_regression_loss_curve.png",
        dpi=150,
    )
    plt.close()

    # Sklearn Linear Regression with Day17 preprocessor.
    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                joblib.load(PREPROCESSOR_PATH),
            ),
            ("regressor", LinearRegression()),
        ]
    )

    linear_model.fit(
        X_train_original,
        y_train,
    )

    linear_predictions = linear_model.predict(
        X_test_original
    )

    # Ridge Regression.
    ridge_model = Pipeline(
        steps=[
            (
                "preprocessor",
                joblib.load(PREPROCESSOR_PATH),
            ),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    ridge_model.fit(
        X_train_original,
        y_train,
    )

    ridge_predictions = ridge_model.predict(
        X_test_original
    )

    # Lasso Regression.
    lasso_model = Pipeline(
        steps=[
            (
                "preprocessor",
                joblib.load(PREPROCESSOR_PATH),
            ),
            (
                "regressor",
                Lasso(
                    alpha=0.01,
                    max_iter=10000,
                ),
            ),
        ]
    )

    lasso_model.fit(
        X_train_original,
        y_train,
    )

    lasso_predictions = lasso_model.predict(
        X_test_original
    )

    # Calculate metrics from scratch.
    scratch_metrics = regression_metrics(
        y_test,
        scratch_predictions,
    )

    linear_metrics = regression_metrics(
        y_test,
        linear_predictions,
    )

    ridge_metrics = regression_metrics(
        y_test,
        ridge_predictions,
    )

    lasso_metrics = regression_metrics(
        y_test,
        lasso_predictions,
    )

    # Validate metrics against sklearn.
    sklearn_linear_mae = mean_absolute_error(
        y_test,
        linear_predictions,
    )

    sklearn_linear_mse = mean_squared_error(
        y_test,
        linear_predictions,
    )

    sklearn_linear_rmse = np.sqrt(
        sklearn_linear_mse
    )

    sklearn_linear_r2 = r2_score(
        y_test,
        linear_predictions,
    )

    print("\nRegression Metrics Validation:")
    print("Scratch MAE:", linear_metrics["MAE"])
    print("Sklearn MAE:", sklearn_linear_mae)
    print("Scratch MSE:", linear_metrics["MSE"])
    print("Sklearn MSE:", sklearn_linear_mse)
    print("Scratch RMSE:", linear_metrics["RMSE"])
    print("Sklearn RMSE:", sklearn_linear_rmse)
    print("Scratch R2:", linear_metrics["R2"])
    print("Sklearn R2:", sklearn_linear_r2)

    # Model comparison table.
    results = pd.DataFrame(
        [
            {
                "Model": "From-Scratch",
                "MAE": scratch_metrics["MAE"],
                "RMSE": scratch_metrics["RMSE"],
                "R2": scratch_metrics["R2"],
            },
            {
                "Model": "Sklearn LinearRegression",
                "MAE": linear_metrics["MAE"],
                "RMSE": linear_metrics["RMSE"],
                "R2": linear_metrics["R2"],
            },
            {
                "Model": "Ridge",
                "MAE": ridge_metrics["MAE"],
                "RMSE": ridge_metrics["RMSE"],
                "R2": ridge_metrics["R2"],
            },
            {
                "Model": "Lasso",
                "MAE": lasso_metrics["MAE"],
                "RMSE": lasso_metrics["RMSE"],
                "R2": lasso_metrics["R2"],
            },
        ]
    )

    print("\nModel Comparison:")
    print(results.to_string(index=False))

    # Compare from-scratch coefficients with sklearn.
    sklearn_coefficients = np.concatenate(
        [
            [
                linear_model.named_steps[
                    "regressor"
                ].intercept_
            ],
            linear_model.named_steps[
                "regressor"
            ].coef_,
        ]
    )

    maximum_difference = np.max(
        np.abs(
            scratch_coefficients
            - sklearn_coefficients
        )
    )

    print("\nCoefficient Comparison:")
    print(
        "Maximum coefficient difference:",
        maximum_difference,
    )

    # Compare Ridge and Lasso regularization effects.
    ridge_coefficients = ridge_model.named_steps[
        "regressor"
    ].coef_

    lasso_coefficients = lasso_model.named_steps[
        "regressor"
    ].coef_

    print("\nRegularization Coefficients:")
    print(
        "Ridge coefficient norm:",
        np.linalg.norm(ridge_coefficients),
    )

    print(
        "Lasso non-zero coefficients:",
        np.count_nonzero(lasso_coefficients),
    )

    # Residual and predicted-vs-actual plots.
    residual_plot_path = (
        CHARTS_DIR
        / "linear_regression_residuals.png"
    )

    plot_residuals(
        y_test,
        linear_predictions,
        str(residual_plot_path),
    )

    print(
        "\nGradient Descent Final Loss:",
        loss_history[-1],
    )

    print(
        "Loss curve saved:",
        CHARTS_DIR
        / "linear_regression_loss_curve.png",
    )

    print(
        "Residual plot saved:",
        residual_plot_path,
    )