
"""Calculate simple baseline metrics for Epic 5."""

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epic5_advanced_ml.project_final_ml_pipeline import MLPipelineRunner


RANDOM_STATE = 42
OUTPUT_DIR = Path("epic5_advanced_ml/outputs")


def run_baseline(task):
    """Evaluate a dummy model using the existing split and CV setup."""
    runner = MLPipelineRunner(
        task=task,
        output_dir=OUTPUT_DIR / task,
    )

    if task == "classification":
        model = DummyClassifier(strategy="most_frequent")
        scoring = "f1"
    else:
        model = DummyRegressor(strategy="mean")
        scoring = "neg_root_mean_squared_error"

    pipeline = runner._build_pipeline(model)

    scores = cross_val_score(
        pipeline,
        runner.X_train,
        runner.y_train,
        cv=runner.cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    pipeline.fit(runner.X_train, runner.y_train)
    predictions = pipeline.predict(runner.X_test)

    results = {
        "task": task,
        "model": type(model).__name__,
        "cv_mean": float(
            np.mean(scores if task == "classification" else -scores)
        ),
        "cv_std": float(np.std(scores)),
    }

    if task == "classification":
        results["test_accuracy"] = float(
            accuracy_score(runner.y_test, predictions)
        )
        results["test_f1"] = float(
            f1_score(
                runner.y_test,
                predictions,
                zero_division=0,
            )
        )
    else:
        results["test_rmse"] = float(
            np.sqrt(
                mean_squared_error(runner.y_test, predictions)
            )
        )
        results["test_r2"] = float(
            r2_score(runner.y_test, predictions)
        )

    output_dir = OUTPUT_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "baseline_metrics.json"
    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    run_baseline("classification")
    run_baseline("regression")