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
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common.ml_utils import build_day17_preprocessor, load_dataset

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5


def build_leaderboard(results: dict) -> pd.DataFrame:
    """Build and sort a leaderboard from model result dictionaries."""
    if not results:
        return pd.DataFrame()

    task = results.get("task")
    rows = []

    for name, result in results.items():
        if name == "task" or not isinstance(result, dict):
            continue

        row = dict(result)
        row.setdefault("model", name)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    leaderboard = pd.DataFrame(rows)

    if "cv_mean" not in leaderboard.columns:
        raise ValueError("Model results must include cv_mean.")

    if task is None:
        task = (
            "regression"
            if {"rmse", "mae", "r2"}.intersection(leaderboard.columns)
            else "classification"
        )

    return leaderboard.sort_values(
        "cv_mean",
        ascending=(task == "regression"),
    ).reset_index(drop=True)


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
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

        # Build preprocessing from training data only.
        # The Pipeline fits it separately inside every CV fold.
        self.preprocessor = build_day17_preprocessor(
            self.X_train
        )

        self.cv = self._build_cv()

        self.models = {}
        self.leaderboard = pd.DataFrame()
        self.best_model = None
        self.best_model_name = None
        self.final_cv_results = {}
        self.tuning_details = {}

    def _build_cv(self):
        """Choose a reproducible validation strategy for the task."""
        if self.task == "classification":
            return StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE,
            )

        return KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    def _build_pipeline(self, estimator):
        """Combine preprocessing and an estimator in one pipeline."""
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
                random_state=RANDOM_STATE,
            )
            knn = KNeighborsClassifier(n_neighbors=5)
            svm = SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            )
            tree = DecisionTreeClassifier(
                random_state=RANDOM_STATE
            )
            forest = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            boosting = GradientBoostingClassifier(
                random_state=RANDOM_STATE
            )

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
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                    cv=3,
                ),
            }

        linear = LinearRegression()
        knn = KNeighborsRegressor(n_neighbors=5)
        svm = SVR(kernel="rbf")
        tree = DecisionTreeRegressor(
            random_state=RANDOM_STATE
        )
        forest = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        boosting = GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )

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
        """Return the scoring metric used for model selection."""
        if self.task == "classification":
            return "f1"

        return "neg_root_mean_squared_error"

    def _cross_validate(self, model):
        """Return fold scores and validation stability statistics."""
        scores = cross_val_score(
            clone(model),
            self.X_train,
            self.y_train,
            cv=self.cv,
            scoring=self._primary_metric(),
            error_score="raise",
            n_jobs=-1,
        )

        # sklearn returns negative RMSE for this regression scorer.
        values = -scores if self.task == "regression" else scores

        return {
            "cv_mean": float(np.mean(values)),
            "cv_std": float(np.std(values)),
            "fold_scores": [
                float(value) for value in values
            ],
        }

    def _evaluate(self, model):
        """Evaluate a fitted model on the held-out test set."""
        predictions = model.predict(self.X_test)

        if self.task == "regression":
            mse = mean_squared_error(
                self.y_test,
                predictions,
            )

            return {
                "mae": float(
                    mean_absolute_error(
                        self.y_test,
                        predictions,
                    )
                ),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(
                    r2_score(self.y_test, predictions)
                ),
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
            probabilities = model.predict_proba(
                self.X_test
            )[:, 1]

            metrics["roc_auc"] = float(
                roc_auc_score(
                    self.y_test,
                    probabilities,
                )
            )

        elif hasattr(model, "decision_function"):
            scores = model.decision_function(self.X_test)

            metrics["roc_auc"] = float(
                roc_auc_score(self.y_test, scores)
            )

        return metrics

    def run_all_models(self):
        """Train candidates and create a CV-based leaderboard."""
        rows = []
        candidates = self._build_candidates()

        self.models = {}
        failures = {}

        for name, estimator in candidates.items():
            try:
                pipeline = self._build_pipeline(estimator)

                start = time.perf_counter()
                pipeline.fit(self.X_train, self.y_train)
                training_time = time.perf_counter() - start

                # Evaluate using training-only cross-validation.
                cv_results = self._cross_validate(pipeline)

                # Test metrics are reported, not used for selection.
                test_metrics = self._evaluate(pipeline)

                row = {
                    "model": name,
                    **cv_results,
                    "training_time_seconds": float(training_time),
                    **test_metrics,
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

        result_mapping = {
            row["model"]: row for row in rows
        }
        result_mapping["task"] = self.task

        self.leaderboard = build_leaderboard(result_mapping)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard.to_csv(
            self.output_dir / "leaderboard.csv",
            index=False,
        )

        return self.leaderboard

    def tune_top_candidates(self, n=2):
        """Tune the top n models using RandomizedSearchCV."""
        if n < 1:
            raise ValueError("n must be at least 1.")

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
            pipeline = self._build_pipeline(candidates[name])
            params = param_distributions.get(name, {})

            try:
                if not params:
                    pipeline.fit(self.X_train, self.y_train)
                    tuned_models[name] = pipeline

                    self.tuning_details[name] = {
                        "best_params": {},
                        "best_cv_score": None,
                        "tuning": "No hyperparameter search configured",
                    }
                    continue

                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=params,
                    n_iter=min(5, max(1, len(params))),
                    cv=self.cv,
                    scoring=self._primary_metric(),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    error_score="raise",
                )

                search.fit(self.X_train, self.y_train)
                tuned_models[name] = search.best_estimator_

                # Record the best parameters and CV score.
                self.tuning_details[name] = {
                    "best_params": search.best_params_,
                    "best_cv_score": float(search.best_score_),
                }

            except Exception:
                logger.exception(
                    "Skipping failed tuning for model: %s",
                    name,
                )

        if not tuned_models:
            raise RuntimeError("No candidate model could be tuned.")

        self.models.update(tuned_models)
        tuned_rows = []

        for name, model in tuned_models.items():
            cv_results = self._cross_validate(model)

            tuned_rows.append({
                "model": name,
                **cv_results,
                "training_time_seconds": None,
                **self._evaluate(model),
            })

        tuned_board = pd.DataFrame(tuned_rows)
        tuned_board = tuned_board.sort_values(
            "cv_mean",
            ascending=(self.task == "regression"),
        ).reset_index(drop=True)

        self.leaderboard = pd.concat(
            [
                self.leaderboard[
                    ~self.leaderboard["model"].isin(tuned_models)
                ],
                tuned_board,
            ],
            ignore_index=True,
        )

        self.leaderboard = self.leaderboard.sort_values(
            "cv_mean",
            ascending=(self.task == "regression"),
        ).reset_index(drop=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard.to_csv(
            self.output_dir / "leaderboard.csv",
            index=False,
        )

        return self.leaderboard

    def cross_validate_final(self, model=None):
        """Cross-validate the selected model and record fold scores."""
        if model is None:
            if self.leaderboard.empty:
                self.run_all_models()

            model_name = str(self.leaderboard.iloc[0]["model"])
            model = self.models[model_name]

        elif isinstance(model, str):
            model_name = model
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
            model = self.models[model_name]

        else:
            model_name = self.best_model_name or "final_model"

        self.final_cv_results = {
            "model": model_name,
            **self._cross_validate(model),
        }
        return self.final_cv_results

    def load_best(self, output_dir=None):
                """Load a previously saved best model without retraining."""
                if output_dir is not None:
                    self.output_dir = Path(output_dir)

                model_path = self.output_dir / "model.joblib"
                metrics_path = self.output_dir / "metrics.json"
                leaderboard_path = self.output_dir / "leaderboard.csv"

                if not model_path.is_file():
                    raise FileNotFoundError(
                        f"Saved model not found: {model_path}. "
                        "Run training and save the model first."
                    )

                if not metrics_path.is_file():
                    raise FileNotFoundError(
                        f"Saved metrics not found: {metrics_path}."
                    )

                self.best_model = joblib.load(model_path)

                with metrics_path.open("r", encoding="utf-8") as file:
                    metrics = json.load(file)

                if metrics.get("task") != self.task:
                    raise ValueError(
                        f"Saved model task is {metrics.get('task')!r}, "
                        f"but requested task is {self.task!r}."
                    )

                self.best_model_name = metrics.get("best_model", "best_model")
                self.models[self.best_model_name] = self.best_model
                self.final_cv_results = metrics.get("cross_validation", {})
                self.tuning_details = metrics.get("tuning_details", {})

                if leaderboard_path.is_file():
                    self.leaderboard = pd.read_csv(leaderboard_path)

                return self.best_model

    def save_best(self, output_dir=None):
        """Save the selected model, metrics, leaderboard, and report."""
        if output_dir is not None:
            self.output_dir = Path(output_dir)

        if self.leaderboard.empty:
            self.run_all_models()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        best_name = str(self.leaderboard.iloc[0]["model"])
        best_model = self.models[best_name]

        self.best_model_name = best_name
        self.best_model = best_model

        model_path = self.output_dir / "model.joblib"
        joblib.dump(best_model, model_path)

        test_metrics = self._evaluate(best_model)
        cv_results = self.cross_validate_final(best_model)

        # Keep metrics at the top level for simple consumers/tests.
        metrics = {
            "task": self.task,
            "best_model": best_name,
            "tuning_details": self.tuning_details,
            **test_metrics,
            "test_metrics": test_metrics,
            "cross_validation": cv_results,
            "validation_strategy": (
                "StratifiedKFold"
                if self.task == "classification"
                else "KFold"
            ),
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "leakage_checks": {
                "preprocessor_created_from_training_data": True,
                "preprocessing_inside_pipeline": True,
                "test_set_used_for_model_selection": False,
            },
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
            "# Model Selection and Validation Report",
            "",
            f"Task: {self.task}",
            f"Selected model: {best_name}",
            "",
            "## Validation strategy",
            "",
            f"- Method: {metrics['validation_strategy']}",
            f"- Folds: {N_SPLITS}",
            f"- Random state: {RANDOM_STATE}",
            f"- Selection metric: {self._primary_metric()}",
            "",
            "## Leakage checks",
            "",
            "- The preprocessor is initialized using training data only.",
            "- Preprocessing is fitted inside the cross-validation pipeline.",
            "- The held-out test set is not used to rank candidate models.",
            "",
            "## Hyperparameter Tuning",
            "",
            "```json",
            json.dumps(self.tuning_details, indent=4),
            "```",
            "",
            "## Leaderboard",
            "",
            self.leaderboard.to_string(index=False),
            "",
            "## Held-out test metrics",
            "",
        ]

        for metric, value in test_metrics.items():
            report.append(f"- **{metric}:** {value:.6f}")

        report.extend([
            "",
            "## Cross-validation stability",
            "",
            f"- Mean: {cv_results['cv_mean']:.6f}",
            f"- Standard deviation: {cv_results['cv_std']:.6f}",
            f"- Fold scores: {cv_results['fold_scores']}",
            "",
            "A lower standard deviation indicates more consistent fold scores. "
            "Review the fold scores and the problem context before concluding "
            "that a model is stable.",
            "",
        ])

        (self.output_dir / "MODEL_SELECTION.md").write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        return model_path
