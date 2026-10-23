# Model Card — Model Validation Capstone

## 1. Purpose
This document describes the selected classification and regression models, their evaluation results, intended use, limitations, and monitoring signals.

## 2. Datasets
- Classification: Breast Cancer Wisconsin dataset from scikit-learn.
- Target: Binary classification of malignant and benign cases.
- Regression: California Housing dataset from scikit-learn.
- Target: Median house value.

## 3. Model Selection
Models were compared using 5-fold cross-validation.
- Classification: Logistic Regression selected based on the highest cross-validation F1 score.
- Regression: Random Forest selected based on the lowest cross-validation RMSE.
- The holdout results below are taken from the existing evaluation records. The holdout set must not be used repeatedly to select or tune models.

## 4. Evaluation Results

### Classification — Logistic Regression
- Cross-validation F1: 0.986116
- Cross-validation F1 standard deviation: 0.004237
- Holdout accuracy: 0.973684
- Holdout precision: 0.972603
- Holdout recall: 0.986111
- Holdout F1: 0.979310
- Holdout ROC-AUC: 0.995701

### Regression — Random Forest
- Cross-validation RMSE: 0.517336
- Cross-validation RMSE standard deviation: 0.011246
- Holdout MAE: 0.331291
- Holdout MSE: 0.258856
- Holdout RMSE: 0.508779
- Holdout R-squared: 0.802461

## 5. Intended Use
- Classification: Educational experimentation and research using the specified dataset.
- Regression: Educational experimentation and analysis of California housing data.
- These models are not intended to make decisions independently in real-world high-impact situations.

## 6. Inference Requirements
- Load the saved pipeline from the relevant model output directory.
- Supply a pandas DataFrame containing the expected feature names.
- Classification requires the 30 original Breast Cancer dataset feature columns.
- Regression requires the 12 expected columns, including the engineered features.
- Input columns and their meanings must match the training pipeline.

## 7. Limitations
- Results are specific to the datasets and evaluation setup used.
- Cross-validation and holdout performance may not represent performance on new data.
- A limited hyperparameter search may miss better configurations.
- Dataset bias, data quality issues, and distribution changes may affect predictions.
- Classification predictions are not a substitute for professional medical assessment.
- Housing estimates may not reflect current prices or individual property values.
- Regression performance can vary across locations and housing market conditions.

## 8. Monitoring Signals
- Monitor missing, unexpected, or incorrectly typed input features.
- Monitor changes in feature distributions and prediction distributions.
- When reliable ground-truth labels become available, monitor classification precision, recall, F1, and confusion matrix.
- For regression, monitor MAE, RMSE, and residual patterns.
- Investigate performance degradation and data drift before deciding whether retraining is needed.

## 9. Reproducibility
- Saved pipelines: `day4_outputs/classification/model.joblib` and `day4_outputs/regression/model.joblib`.
- Comparison and evaluation records are stored in the corresponding output directories.
- Use the project's documented environment and execution instructions to reproduce supported checks.
- Do not rerun holdout evaluation simply to reproduce the existing recorded metrics.

## 10. Verification Evidence
- Classification pipeline produced predictions for two correctly named input rows.
- Classification pipeline rejected an input containing unexpected column names.
- Regression pipeline produced predictions for two rows after applying the project's feature-engineering function.
