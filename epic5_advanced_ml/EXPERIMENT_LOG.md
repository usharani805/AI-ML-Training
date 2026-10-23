# Machine Learning Training - Experiment Log

**Project:** Epic 5 Advanced ML
**Training Period:** 12-16 October 2026

## Day 1 - Baseline Review

**Status:** Completed.

**Objective:** Establish simple baseline scores before comparing advanced ML models.

### Classification - Breast Cancer
- Dataset: 569 rows, 30 features.
- Train/test split: 455 / 114.
- Split strategy: 80/20, `random_state=42`, stratified.
- Baseline model: `DummyClassifier(strategy="most_frequent")`.
- CV strategy: 5-fold `StratifiedKFold`.
- CV F1 mean: 0.770270.
- CV F1 standard deviation: 0.000000.
- Test accuracy: 0.631579.
- Test F1: 0.774194.

### Regression - California Housing
- Dataset: 20,640 rows, 8 original features before feature engineering.
- Train/test split: 16,512 / 4,128.
- Split strategy: 80/20, `random_state=42`.
- Baseline model: `DummyRegressor(strategy="mean")`.
- CV strategy: 5-fold `KFold`.
- CV RMSE mean: 1.156172.
- CV RMSE standard deviation: 0.012163.
- Test RMSE: 1.144856.
- Test R2: -0.000219.

### Baseline conclusion
The dummy models provide reference scores for later model comparisons. The classification baseline predicts the majority class, while the regression baseline predicts the training target mean. The negative regression R2 indicates that the baseline does not explain the test target variance and provides a reference for evaluating trained regression models.

**Output files:**
- `epic5_advanced_ml/outputs/classification/baseline_metrics.json`
- `epic5_advanced_ml/outputs/regression/baseline_metrics.json`

## Day 2 - Validation Strategy

### Objective
Select an appropriate validation strategy and inspect
data leakage risks and model result stability.

### Validation Configuration

| Setting | Classification | Regression |
|---|---|---|
| Validation method | StratifiedKFold | KFold |
| Number of folds | 5 | 5 |
| Shuffle | True | True |
| Random state | 42 | 42 |
| Dataset | Breast Cancer | California Housing |
| CV scoring metric | F1 score | Negative RMSE (reported as positive RMSE) |

### Cross-Validation Results

| Result | Classification | Regression |
|---|---:|---:|
| CV mean | 0.984525 | 0.517237 |
| CV standard deviation | 0.012517 | 0.003619 |
| Fold 1 | 0.973913 | 0.517507 |
| Fold 2 | 1.000000 | 0.520458 |
| Fold 3 | 0.966102 | 0.512647 |
| Fold 4 | 0.991304 | 0.513703 |
| Fold 5 | 0.991304 | 0.521871 |

### Leakage Review

- Target columns are separated from the input features.
- Reviewed regression feature engineering; the inspected
  engineered features do not use the target column.
- Preprocessing is included in the model pipeline.
- Preprocessing is expected to be fitted independently
  within each cross-validation training fold.
- These checks are based on code review; they do not prove
  that every possible source of leakage has been eliminated.

### Stability Review

- Classification fold scores show some variation.
- Regression fold scores are relatively close together.
- The reported CV standard deviations are recorded for
  comparison with future experiments.

### Test Results

- Test command: `pytest epic5_advanced_ml/test_project_epic5.py -q`
- Result: 35 passed, 8 warnings.
- Python compilation was checked successfully.

### Output Files

- `epic5_advanced_ml/outputs/classification/metrics.json`
- `epic5_advanced_ml/outputs/regression/metrics.json`

### Day 2 Status

Validation configuration and CV results reviewed.
Experiment log updated and reviewed.

## Day 3 - Model Comparison

**Status:** Completed.

### Objective
Compare classification and regression models using consistent cross-validation strategies and evaluation metrics.

### Classification Comparison

Dataset: Breast Cancer
Validation: 5-fold StratifiedKFold, shuffle=True, random_state=42
Primary metric: F1 score

| Model | CV Mean | CV Std | Test F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.984525 | 0.012517 | 0.979310 |
| Stacking | 0.981018 | 0.019954 | 0.986301 |
| SVM | 0.977292 | 0.014038 | 0.986111 |
| KNN | 0.973999 | 0.016362 | 0.965517 |
| Gradient Boosting | 0.965030 | 0.010638 | 0.965986 |
| Random Forest | 0.963408 | 0.018367 | 0.965517 |
| Decision Tree | 0.928176 | 0.014981 | 0.928571 |

Observation: Logistic Regression has the highest CV mean F1 score. Stacking has the highest test F1 score. Model selection should prioritize the predefined CV metric and consider test performance as supporting evidence.

### Regression Comparison

Dataset: California Housing
Validation: 5-fold KFold, shuffle=True, random_state=42
Primary metric: RMSE (lower is better)

| Model | CV RMSE | CV Std | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|
| Random Forest | 0.517237 | 0.003619 | 0.508779 | 0.802461 |
| Gradient Boosting | 0.519753 | 0.006242 | 0.527564 | 0.787606 |
| Stacking | 0.585983 | 0.008537 | 0.578058 | 0.745003 |
| SVM | 0.602918 | 0.010623 | 0.609606 | 0.716410 |
| KNN | 0.658935 | 0.013510 | 0.672138 | 0.655246 |
| Linear Regression | 0.681365 | 0.030302 | 0.675556 | 0.651730 |
| Decision Tree | 0.742857 | 0.011924 | 0.705529 | 0.620142 |

Observation: Random Forest has the lowest CV RMSE and test RMSE, and the highest test R2 among the compared regression models.

### Day 3 Conclusion

- Classification: Logistic Regression leads on the primary CV F1 metric.
- Regression: Random Forest leads on the primary CV RMSE metric.
- The existing validation strategy and random state are retained for comparison.
- Hyperparameter tuning and further diagnostics are reserved for Day 4.

### Preprocessing Comparison

**Objective:** Compare three preprocessing methods while keeping the selected model and cross-validation strategy consistent for each task.

| Task | Model | Preprocessing | CV Mean | CV Std |
|---|---|---|---:|---:|
| Classification | Logistic Regression | Day 17 Existing | 0.979434 | 0.012725 |
| Classification | Logistic Regression | Standard Scaler | 0.979434 | 0.012725 |
| Classification | Logistic Regression | MinMax Scaler | 0.972889 | 0.014933 |
| Regression | Random Forest | Day 17 Existing | 0.507727 | 0.009910 |
| Regression | Random Forest | Standard Scaler | 0.507575 | 0.009806 |
| Regression | Random Forest | MinMax Scaler | 0.507215 | 0.009435 |

**Observations:**
- Classification: Day 17 Existing and Standard Scaler produced the same CV mean F1 score.
- Regression: MinMax Scaler produced the lowest CV RMSE in this experiment.
- The differences are small. These results apply to the specific models and experimental setup used; they do not establish a universally best preprocessing method.

**Output:** `epic5_advanced_ml/outputs/preprocessing_comparison.csv`

## Day 4 - Tuning and Diagnostics

**Status:** Results reviewed; documentation updated.

### Objective
Compare tuned models against dummy baselines and the Day 3 model comparison using the existing five-fold cross-validation strategy.

### Baseline Results

| Task | Baseline Model | CV Mean | CV Std | Test Metric |
|---|---|---:|---:|---|
| Classification | DummyClassifier | F1: 0.770270 | 0.000000 | Accuracy: 0.631579; F1: 0.774194 |
| Regression | DummyRegressor | RMSE: 1.156172 | 0.012163 | RMSE: 1.144856; R2: -0.000219 |

### Classification Tuning Results

| Model | Best CV F1 | CV Std | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.986116 | 0.004237 | 0.979310 | 0.995701 |
| SVM | 0.981015 | Not recorded here | 0.986111 | 0.993717 |

Best parameters:
- Logistic Regression: `C=0.1`
- SVM: `kernel=linear`, `C=0.1`

Observation: Logistic Regression achieved the higher reported CV F1 score. SVM achieved the higher test F1 score. Model selection follows the predefined CV metric.

### Regression Tuning Results

| Model | Best CV RMSE | CV Std | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|
| Random Forest | 0.517336 | 0.011246 | 0.508779 | 0.802461 |
| Gradient Boosting | 0.521121 | 0.010958 | 0.527564 | 0.787606 |

Best parameters:
- Random Forest: `n_estimators=100`, `max_depth=None`
- Gradient Boosting: `n_estimators=150`, `max_depth=5`, `learning_rate=0.03`

Observation: Random Forest achieved the lower CV RMSE and test RMSE among these tuned regression models.

### Validation and Diagnostics

- Classification uses five-fold StratifiedKFold.
- Regression uses five-fold KFold.
- Shuffle is enabled and random state is 42.
- Preprocessing is included in the model pipeline.
- Code review checks target separation and inspected feature engineering for potential leakage.
- These code checks do not prove that all possible leakage has been eliminated.
- Python compilation succeeded.
- Epic 5 test suite: 70 passed, 16 warnings.

### Output Files

- `epic5_advanced_ml/outputs/classification/baseline_metrics.json`
- `epic5_advanced_ml/outputs/regression/baseline_metrics.json`
- `epic5_advanced_ml/day4_outputs/classification/`
- `epic5_advanced_ml/day4_outputs/regression/`

### Day 4 Conclusion

The tuned Logistic Regression model leads classification on the reported CV F1 metric. Random Forest leads regression on the reported CV RMSE metric. Both tuned models outperform their dummy baselines on the selected CV metrics.

## Day 5 - Results and Reproducibility

**Status:** Documentation created and metrics reviewed.

### Objective
Document the selected models, evaluation results, and steps needed to reproduce the experiments.

### Results Review
- Classification: Logistic Regression (`C=0.1`), CV F1 = 0.986116, test F1 = 0.979310, test ROC-AUC = 0.995701.
- Regression: Random Forest (`n_estimators=100`, `max_depth=None`), CV RMSE = 0.517336, test RMSE = 0.508779, test RÂ² = 0.802461.
- Classification and regression metrics were checked against the Day 4 `metrics.json` files.

### Documentation
- `README.md` documents the project, validation strategy, selected models, and run commands.
- `EVALUATION_REPORT.md` records the selected models and evaluation metrics.
- `EXPERIMENT_LOG.md` records the work and results for Days 1-5.

### Reproducibility
- Random state: 42.
- Cross-validation: 5 folds.
- Classification: StratifiedKFold.
- Regression: KFold.
- Detailed model artifacts and tuning results are stored in `day4_outputs/`.

### Verification Results
- Python compilation: no errors.
- Epic 5 test suite: 35 passed, 7 warnings.
- Classification and regression metrics were checked against the saved `metrics.json` files.
- Git changes were reviewed before selecting files for commit.
- The regression model artifact is approximately 137.86 MB and is excluded from this commit.

## Reproducibility Verification — 22 October 2026

### Execution Details
- Code revision before this verification: 8a259fe
- Dataset: Breast Cancer classification and California Housing regression.
- Random state: 42.
- Cross-validation: 5 folds; StratifiedKFold for classification and KFold for regression.
- Preprocessing: included in the model pipeline.
- Baseline comparison files: `day4_outputs/`.
- Rerun artifacts: `day4_reproducibility_check/`.

### Classification Rerun
- Best model: Logistic Regression.
- Accuracy: 0.973684.
- Precision: 0.972603.
- Recall: 0.986111.
- F1: 0.979310.
- ROC-AUC: 0.995701.
- Comparison: all listed metrics match the saved Day 4 results.

### Regression Rerun
- Best model: Random Forest.
- MAE: 0.331291.
- MSE: 0.258856.
- RMSE: 0.508779.
- R²: 0.802461.
- Comparison: metrics match within floating-point tolerance. MAE differs only in the final floating-point digits.

### Verification Evidence
- Python compilation: successful.
- Epic 5 test suite: 37 passed, 7 warnings.
- Rerun metrics were compared with the saved Day 4 metrics.
- Model artifacts were generated in the reproducibility-check output directories.
- Limitation: matching metrics do not independently prove that every possible source of data leakage has been eliminated.

## Model Validation Capstone - Inference Verification (23 Oct 2026)

- Classification: Loaded the saved Logistic Regression pipeline and generated predictions for two correctly named input rows.
- Negative case: The classification pipeline rejected input with unexpected column names.
- Regression: Loaded the saved Random Forest pipeline, applied the existing feature-engineering function, and generated predictions for two rows.
- Holdout metrics: Reused existing evaluation records; no new holdout evaluation was performed during these inference checks.
- Limitation: These checks verify basic inference behavior only; they do not establish production readiness.
