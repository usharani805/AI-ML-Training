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