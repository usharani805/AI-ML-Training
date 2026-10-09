
"""Final end-to-end machine learning pipeline for Epic 5."""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common.ml_utils import (
    build_day17_preprocessor,
    load_dataset,
)

logger = logging.getLogger(__name__)


class MLPipelineRunner:
    """Train, compare, tune, validate, and save ML models."""

    def __init__(
        self,
        task,
        output_dir="epic5_advanced_ml/day25_outputs",
    ):
        if task not in {"classification", "regression"}:
            raise ValueError(
                "task must be 'classification' or 'regression'"
            )

        self.task = task
        self.output_dir = Path(output_dir)

        self.X, self.y = load_dataset(task)
        self.preprocessor = build_day17_preprocessor(self.X)

        stratify = self.y if task == "classification" else None

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        self.models = {}
        self.leaderboard = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.final_cv_results = {}

    def _build_pipeline(self, estimator):
        """Combine Day 17 preprocessing with an estimator."""
        return Pipeline(
            steps=[
                ("preprocessor", clone(self.preprocessor)),
                ("model", clone(estimator)),
            ]
        )

    def _build_candidates(self):
        """Return candidate estimators for the selected task."""
        if self.task == "classification":
            logistic = LogisticRegression(
                max_iter=2000,
                random_state=42,
            )
            knn = KNeighborsClassifier(n_neighbors=5)
            svm = SVC(
                kernel="rbf",
                probability=True,
                random_state=42,
            )
            tree = DecisionTreeClassifier(random_state=42)
            forest = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
            boosting = GradientBoostingClassifier(random_state=42)

            return {
                "logistic_regression": logistic,
                "knn": knn,
                "svm": svm,
                "decision_tree": tree,
                "random_forest": forest,
                "gradient_boosting": boosting,
                "stacking": StackingClassifier(
                    estimators=[
                        ("logistic", clone(logistic)),
                        ("knn", clone(knn)),
                        ("tree", clone(tree)),
                    ],
                    final_estimator=LogisticRegression(
                        max_iter=2000
                    ),
                    cv=3,
                ),
            }

        linear = LinearRegression()
        knn = KNeighborsRegressor(n_neighbors=5)
        svm = SVR(kernel="rbf")
        tree = DecisionTreeRegressor(random_state=42)
        forest = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        boosting = GradientBoostingRegressor(random_state=42)

        return {
            "linear_regression": linear,
            "knn": knn,
            "svm": svm,
            "decision_tree": tree,
            "random_forest": forest,
            "gradient_boosting": boosting,
            "stacking": StackingRegressor(
                estimators=[
                    ("linear", clone(linear)),
                    ("knn", clone(knn)),
                    ("tree", clone(tree)),
                ],
                final_estimator=LinearRegression(),
                cv=3,
            ),
        }

    def _primary_metric(self):
        """Return the cross-validation scoring metric."""
        if self.task == "classification":
            return "f1"
        return "neg_root_mean_squared_error"

    def _evaluate(self, model):
        """Evaluate a fitted model on the held-out test set."""
        predictions = model.predict(self.X_test)

        if self.task == "regression":
            mse = mean_squared_error(self.y_test, predictions)
            return {
                "mae": float(
                    mean_absolute_error(self.y_test, predictions)
                ),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(self.y_test, predictions)),
            }

        metrics = {
            "accuracy": float(
                accuracy_score(self.y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    self.y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    self.y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    self.y_test,
                    predictions,
                    zero_division=0,
                )
            ),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(self.X_test)[:, 1]
            metrics["roc_auc"] = float(
                roc_auc_score(self.y_test, probabilities)
            )
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(self.X_test)
            metrics["roc_auc"] = float(
                roc_auc_score(self.y_test, scores)
            )

        return metrics

    def run_all_models(self):
        """Train candidates and create a sorted leaderboard."""
        rows = []
        candidates = self._build_candidates()
        self.models = {}
        failures = {}

        for name, estimator in candidates.items():
            try:
                if isinstance(estimator, Pipeline):
                    pipeline = clone(estimator)
                    cv_pipeline = clone(estimator)
                else:
                    pipeline = self._build_pipeline(estimator)
                    cv_pipeline = self._build_pipeline(estimator)

                start = time.perf_counter()
                pipeline.fit(self.X_train, self.y_train)
                training_time = time.perf_counter() - start

                scores = cross_val_score(
                    cv_pipeline,
                    self.X_train,
                    self.y_train,
                    cv=5,
                    scoring=self._primary_metric(),
                    error_score="raise",
                )

                cv_values = (
                    -scores
                    if self.task == "regression"
                    else scores
                )

                row = {
                    "model": name,
                    "cv_mean": float(np.mean(cv_values)),
                    "cv_std": float(np.std(cv_values)),
                    "training_time_seconds": float(training_time),
                    **self._evaluate(pipeline),
                }

                rows.append(row)
                self.models[name] = pipeline
                logger.info("Completed model: %s", name)

            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                logger.exception("Skipping failed model: %s", name)

        if not rows:
            details = "\n".join(
                f"{name}: {error}"
                for name, error in failures.items()
            )
            raise RuntimeError(
                "All model candidates failed. Actual errors:\n"
                + details
            )

        leaderboard = pd.DataFrame(rows)
        leaderboard = leaderboard.sort_values(
            "cv_mean",
            ascending=(self.task == "regression"),
        ).reset_index(drop=True)

        self.leaderboard = leaderboard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(
            self.output_dir / "leaderboard.csv",
            index=False,
        )

        return leaderboard

    def tune_top_candidates(self, n=2):
        """Tune the top n candidates using RandomizedSearchCV."""
        if self.leaderboard.empty:
            self.run_all_models()

        top_names = self.leaderboard["model"].head(n).tolist()
        candidates = self._build_candidates()
        tuned_models = {}

        param_distributions = {
            "logistic_regression": {
                "model__C": [0.1, 1.0, 10.0],
            },
            "knn": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
            },
            "svm": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["linear", "rbf"],
            },
            "decision_tree": {
                "model__max_depth": [None, 3, 5, 10],
            },
            "random_forest": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [None, 5, 10],
            },
            "gradient_boosting": {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.03, 0.1, 0.2],
                "model__max_depth": [2, 3, 5],
            },
            "linear_regression": {},
            "stacking": {},
        }

        for name in top_names:
            estimator = candidates[name]

            if isinstance(estimator, Pipeline):
                pipeline = clone(estimator)
            else:
                pipeline = self._build_pipeline(estimator)

            params = param_distributions.get(name, {})

            if not params:
                tuned_models[name] = pipeline.fit(
                    self.X_train,
                    self.y_train,
                )
                continue

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=params,
                n_iter=min(5, max(1, len(params))),
                cv=5,
                scoring=self._primary_metric(),
                random_state=42,
                n_jobs=-1,
                error_score="raise",
            )

            search.fit(self.X_train, self.y_train)
            tuned_models[name] = search.best_estimator_

        if not tuned_models:
            raise RuntimeError("No candidate model could be tuned.")

        tuned_rows = []
        self.models.update(tuned_models)

        for name, model in tuned_models.items():
            scores = cross_val_score(
                clone(model),
                self.X_train,
                self.y_train,
                cv=5,
                scoring=self._primary_metric(),
                error_score="raise",
            )

            cv_values = (
                -scores if self.task == "regression" else scores
            )

            tuned_rows.append({
                "model": name,
                "cv_mean": float(np.mean(cv_values)),
                "cv_std": float(np.std(cv_values)),
                "training_time_seconds": None,
                **self._evaluate(model),
            })

        tuned_board = pd.DataFrame(tuned_rows)
        tuned_board = tuned_board.sort_values(
            "cv_mean",
            ascending=(self.task == "regression"),
        ).reset_index(drop=True)

        self.leaderboard = pd.concat(
            [self.leaderboard, tuned_board],
            ignore_index=True,
        )
        self.leaderboard = self.leaderboard.drop_duplicates(
            subset=["model"],
            keep="last",
        ).sort_values(
            "cv_mean",
            ascending=(self.task == "regression"),
        ).reset_index(drop=True)

        self.leaderboard.to_csv(
            self.output_dir / "leaderboard.csv",
            index=False,
        )

        return self.leaderboard

    def cross_validate_final(self, model=None):
        """Cross-validate the selected model and store the results."""
        if model is None:
            if self.leaderboard.empty:
                self.run_all_models()
            model_name = self.leaderboard.iloc[0]["model"]
            model = self.models[model_name]
        elif isinstance(model, str):
            model_name = model
            if model_name not in self.models:
                raise ValueError(
                    f"Model not found: {model_name}"
                )
            model = self.models[model_name]
        else:
            model_name = getattr(model, "name", "final_model")

        scores = cross_val_score(
            clone(model),
            self.X_train,
            self.y_train,
            cv=5,
            scoring=self._primary_metric(),
            error_score="raise",
        )

        values = (
            -scores if self.task == "regression" else scores
        )

        self.final_cv_results = {
            "model": model_name,
            "cv_mean": float(np.mean(values)),
            "cv_std": float(np.std(values)),
            "fold_scores": [float(value) for value in values],
        }

        return self.final_cv_results

    def save_best(self, output_dir=None):
        """Save the selected model, metrics, leaderboard, and report."""
        if self.leaderboard.empty:
            self.run_all_models()

        if output_dir is not None:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        best_name = str(self.leaderboard.iloc[0]["model"])
        best_model = self.models[best_name]

        self.best_model_name = best_name
        self.best_model = best_model

        model_path = self.output_dir / "model.joblib"
        joblib.dump(best_model, model_path)

        # Keep metric values both at the top level and in test_metrics.
        test_metrics = self._evaluate(best_model)
        cv_results = self.cross_validate_final(best_model)

        metrics = {
            "task": self.task,
            "best_model": best_name,
            **test_metrics,
            "test_metrics": test_metrics,
            "cross_validation": cv_results,
        }

        with (self.output_dir / "metrics.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=4)

        self.leaderboard.to_csv(
            self.output_dir / "leaderboard.csv",
            index=False,
        )

        report = [
            "# Model Selection",
            "",
            f"Task: {self.task}",
            "",
            f"Selected model: {best_name}",
            "",
            "Selection is based on cross-validation performance.",
            "Training time and model interpretability should also "
            "be considered when choosing a model for deployment.",
            "",
            "## Leaderboard",
            "",
            self.leaderboard.to_string(index=False),
            "",
            "## Test Metrics",
            "",
        ]

        for metric, value in test_metrics.items():
            report.append(f"- **{metric}:** {value:.6f}")

        report.extend([
            "",
            "## Cross-Validation",
            "",
            f"- Mean: {cv_results['cv_mean']:.6f}",
            f"- Standard deviation: {cv_results['cv_std']:.6f}",
            "",
        ])

        (self.output_dir / "MODEL_SELECTION.md").write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        return model_path