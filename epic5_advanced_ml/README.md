# Epic 5 Advanced ML Pipeline

## Objective
Compare, tune, evaluate, and select machine learning models using reproducible validation.

## Datasets
- Classification: Breast Cancer (569 rows, 30 features)
- Regression: California Housing (20,640 rows, 8 original features)

## Validation
- Classification: 5-fold StratifiedKFold
- Regression: 5-fold KFold
- Shuffle: True
- Random state: 42
- Classification metric: F1 score (higher is better)
- Regression metric: RMSE (lower is better)

## Selected Models
- Classification: Logistic Regression, C=0.1
- Regression: Random Forest, n_estimators=100, max_depth=None

## Results
Latest tuning results are documented in `EXPERIMENT_LOG.md`.

Detailed metrics and model selection notes are in `day4_outputs/classification/` and `day4_outputs/regression/`.

## Limitations
Results depend on the data, preprocessing, validation setup, and package versions. Leakage checks were based on code review and do not prove that all possible leakage has been eliminated.

## Day 2: Leakage-Safe Tuning and Resource Limits

- **Pipeline:** Preprocessing and model fitting are combined in a scikit-learn `Pipeline` to reduce data leakage during cross-validation.
- **Training-only validation:** `RandomizedSearchCV` uses `self.cv` and fits on `self.X_train` and `self.y_train`. The test set is not used for hyperparameter selection.
- **Reproducibility:** `RANDOM_STATE = 42` is used for randomized search and applicable model settings.
- **Bounded search:** `n_iter=min(5, max(1, len(params)))` limits the number of sampled parameter configurations to at most five.
- **Resource limit:** `RandomizedSearchCV` uses `n_jobs=1` to run one search job at a time.
- **Failure handling:** `error_score="raise"` exposes fitting errors. Failed candidate tuning is logged and skipped. If no candidate succeeds, a `RuntimeError` is raised.
- **Boundary validation:** `tune_top_candidates(n=0)` raises a `ValueError`.
- **Limitations:** A small search budget may miss better hyperparameters. Other estimators and cross-validation operations may use their own parallel processing settings.
