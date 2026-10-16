# Epic 5 - Model Evaluation Report

## 1. Objective
Evaluate classification and regression models using cross-validation, hyperparameter tuning, and test-set metrics.

## 2. Validation Strategy
- Cross-validation: 5 folds
- Random state: 42
- Classification: StratifiedKFold
- Regression: KFold
- Classification primary metric: F1 score (higher is better)
- Regression primary metric: RMSE (lower is better)

## 3. Classification Results
Dataset: Breast Cancer.

- Selected model: Logistic Regression
- Best hyperparameter: C = 0.1
- Cross-validation F1 score: 0.986116
- Test F1 score: 0.979310
- Test ROC-AUC: 0.995701

The selected Logistic Regression model achieved a higher cross-validation F1 score than the tuned SVM candidate (0.981015). Model selection follows the predefined cross-validation metric.

## 4. Regression Results
Dataset: California Housing.

- Selected model: Random Forest Regressor
- Number of trees: 100
- Maximum depth: None
- Cross-validation RMSE: 0.517336
- Test RMSE: 0.508779
- Test R2: 0.802461

Random Forest achieved a lower cross-validation RMSE than Gradient Boosting (0.521121), so it was selected based on the predefined metric.

## 5. Baseline Comparison
Baseline metrics are saved in:
- `outputs/classification/baseline_metrics.json`
- `outputs/regression/baseline_metrics.json`

The experiment log records that both selected models outperform their dummy baselines on the respective cross-validation metrics. Refer to these files for the recorded baseline values.

## 6. Limitations
Results depend on the datasets, preprocessing, validation strategy, and software versions. Code checks help identify potential data leakage but cannot prove that every possible leakage risk has been eliminated.

## 7. Reproducibility
- Random state: 42
- Cross-validation: 5 folds
- Classification: StratifiedKFold
- Regression: KFold
- Detailed tuning results: `EXPERIMENT_LOG.md`
- Metrics, leaderboards, and model selection notes: `day4_outputs/classification/` and `day4_outputs/regression/`

## 8. Run Commands
Run from the repository root with the project Python environment activated:

```powershell
python -m epic5_advanced_ml.train --task classification --tune --output-dir epic5_advanced_ml/day4_outputs/classification
python -m epic5_advanced_ml.train --task regression --tune --output-dir epic5_advanced_ml/day4_outputs/regression
```
