
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from common.ml_utils import load_dataset, build_day17_preprocessor
from epic5_advanced_ml.project_final_ml_pipeline import MLPipelineRunner


def build_alternative_preprocessor(X, scaler):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def compare_preprocessing(task):
    X, y = load_dataset(task)
    runner = MLPipelineRunner(task=task)

    if task == "classification":
        model = LogisticRegression(max_iter=2000, random_state=42)
        scoring = "f1"
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        scoring = "neg_root_mean_squared_error"

    preprocessors = {
        "day17_existing": build_day17_preprocessor(X),
        "standard_scaler_all_numeric": build_alternative_preprocessor(
            X, StandardScaler()
        ),
        "minmax_scaler_all_numeric": build_alternative_preprocessor(
            X, MinMaxScaler()
        ),
    }

    rows = []

    for name, preprocessor in preprocessors.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=runner.cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        mean_score = scores["test_score"].mean()
        std_score = scores["test_score"].std()

        if task == "regression":
            mean_score = -mean_score

        rows.append({
            "task": task,
            "preprocessing": name,
            "cv_mean": mean_score,
            "cv_std": std_score,
            "metric": "F1" if task == "classification" else "RMSE",
        })

        print(
            f"{task} | {name} | "
            f"CV mean={mean_score:.6f} | CV std={std_score:.6f}"
        )

    return rows


def main():
    results = []
    results.extend(compare_preprocessing("classification"))
    results.extend(compare_preprocessing("regression"))

    output_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "preprocessing_comparison.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()