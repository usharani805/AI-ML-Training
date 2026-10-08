import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "epic4_ml" / "data" / "california_housing.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "day24_outputs"


def manual_kfold_cv(X, y, model, k: int, task: str) -> list[float]:
    """Run manual K-Fold or Stratified K-Fold cross-validation."""

    if task == "classification":
        splitter = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=42,
        )
    elif task == "regression":
        splitter = KFold(
            n_splits=k,
            shuffle=True,
            random_state=42,
        )
    else:
        raise ValueError(
            "task must be 'classification' or 'regression'"
        )

    scores = []

    for train_indices, test_indices in splitter.split(X, y):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        predictions = fold_model.predict(X_test)

        if task == "classification":
            score = accuracy_score(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)

        scores.append(score)

    return scores


def run_grid_search(
    pipeline,
    param_grid: dict,
    X,
    y,
    cv: int,
) -> GridSearchCV:
    """Run GridSearchCV on the given pipeline."""

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    return grid_search


def run_random_search(
    pipeline,
    param_distributions: dict,
    X,
    y,
    n_iter: int,
    cv: int,
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV on the given pipeline."""

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X, y)

    return random_search


def nested_cv_score(
    pipeline,
    param_grid: dict,
    X,
    y,
    outer_k: int,
    inner_k: int,
) -> dict:
    """
    Run nested cross-validation.

    The outer loop gives an unbiased performance estimate.
    The inner loop performs hyperparameter tuning.

    Keeping the outer test fold separate from tuning helps
    avoid optimistic performance estimates.
    """

    outer_cv = KFold(
        n_splits=outer_k,
        shuffle=True,
        random_state=42,
    )

    outer_scores = []

    for train_indices, test_indices in outer_cv.split(X, y):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        inner_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_k,
            scoring="r2",
            n_jobs=-1,
        )

        inner_search.fit(X_train, y_train)

        predictions = inner_search.best_estimator_.predict(X_test)

        score = r2_score(y_test, predictions)
        outer_scores.append(score)

    return {
        "outer_scores": outer_scores,
        "mean_score": sum(outer_scores) / len(outer_scores),
    }


def build_xgboost_pipeline():
    """Build preprocessing and XGBoost model in one pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


PARAM_GRID = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
}


def load_regression_data():
    """Load California Housing regression data."""

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    return X, y


def save_cv_results(search, filename: str):
    """Export cv_results_ to CSV."""

    results_df = pd.DataFrame(search.cv_results_)

    output_path = OUTPUT_DIR / filename
    results_df.to_csv(output_path, index=False)

    return results_df


def plot_parameter_scores(results_df, parameter_name: str):
    """Plot parameter values against mean CV score."""

    column_name = f"param_model__{parameter_name}"

    if column_name not in results_df.columns:
        return

    plot_df = (
        results_df.groupby(column_name)["mean_test_score"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        plot_df[column_name].astype(str),
        plot_df["mean_test_score"],
        marker="o",
    )
    plt.xlabel(parameter_name)
    plt.ylabel("Mean CV R2 Score")
    plt.title(f"{parameter_name} vs CV Score")
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"day24_{parameter_name}_vs_score.png"
    )

    plt.savefig(output_path)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_regression_data()

    pipeline = build_xgboost_pipeline()

    print("Dataset shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nManual K-Fold CV:")

    manual_scores = manual_kfold_cv(
        X,
        y,
        pipeline,
        k=5,
        task="regression",
    )

    print("Manual K-Fold scores:", manual_scores)
    print("Manual K-Fold mean:", sum(manual_scores) / len(manual_scores))

    sklearn_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    print("\nSklearn cross_val_score:")
    print("Sklearn scores:", sklearn_scores.tolist())
    print("Sklearn mean:", sklearn_scores.mean())

    print(
        "\nManual vs sklearn mean difference:",
        abs(
            (sum(manual_scores) / len(manual_scores))
            - sklearn_scores.mean()
        ),
    )

    print("\nRunning GridSearchCV...")

    grid_start = time.perf_counter()

    grid_search = run_grid_search(
        pipeline,
        PARAM_GRID,
        X,
        y,
        cv=5,
    )

    grid_time = time.perf_counter() - grid_start

    print("Grid best parameters:")
    print(grid_search.best_params_)

    print("Grid best score:")
    print(grid_search.best_score_)

    print("Grid search time:")
    print(f"{grid_time:.2f} seconds")

    grid_results = save_cv_results(
        grid_search,
        "day24_grid_cv_results.csv",
    )

    print("\nRunning RandomizedSearchCV...")

    random_start = time.perf_counter()

    random_search = run_random_search(
        pipeline,
        PARAM_GRID,
        X,
        y,
        n_iter=8,
        cv=5,
    )

    random_time = time.perf_counter() - random_start

    print("Random best parameters:")
    print(random_search.best_params_)

    print("Random best score:")
    print(random_search.best_score_)

    print("Random search time:")
    print(f"{random_time:.2f} seconds")

    random_results = save_cv_results(
        random_search,
        "day24_random_cv_results.csv",
    )

    print("\nNested Cross-Validation:")

    nested_results = nested_cv_score(
        pipeline,
        PARAM_GRID,
        X,
        y,
        outer_k=5,
        inner_k=5,
    )

    print("Nested outer fold scores:")
    print(nested_results["outer_scores"])

    print("Nested mean score:")
    print(nested_results["mean_score"])

    comparison = pd.DataFrame(
        [
            {
                "Search": "GridSearchCV",
                "Best Score": grid_search.best_score_,
                "Time Seconds": grid_time,
            },
            {
                "Search": "RandomizedSearchCV",
                "Best Score": random_search.best_score_,
                "Time Seconds": random_time,
            },
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR / "day24_grid_vs_random_comparison.csv",
        index=False,
    )

    print("\nGrid vs Randomized Search:")
    print(comparison)

    for parameter in [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
    ]:
        plot_parameter_scores(
            grid_results,
            parameter,
        )

    print("\nDay24 completed.")


if __name__ == "__main__":
    main()