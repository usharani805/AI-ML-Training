# Epic 4 - Decision Trees, Random Forest & Model Comparison

## Run Instructions

Run the Decision Tree and Random Forest analysis:

```bash
python epic4_ml/day20_trees_and_project.py
```

Run the training CLI for classification:

```bash
python epic4_ml/train.py --task classification --model random_forest --output-dir models
```

Run the training CLI for regression:

```bash
python epic4_ml/train.py --task regression --model random_forest --output-dir models
```

## Datasets

### Classification

* Dataset: Breast Cancer dataset
* Target column: `target`
* Evaluation metric: Accuracy

### Regression

* Dataset: California Housing dataset
* Target column: `MedHouseVal`
* Evaluation metric: R² Score

## Model Comparison

### Classification

| Model               | Train Score | Test Score |
| ------------------- | ----------: | ---------: |
| Baseline            |    0.626374 |   0.631579 |
| Logistic Regression |    0.960440 |   0.964912 |
| Decision Tree       |    1.000000 |   0.912281 |
| Random Forest       |    1.000000 |   0.956140 |

### Regression

| Model             | Train Score | Test Score |
| ----------------- | ----------: | ---------: |
| Baseline          |    0.000000 |  -0.000219 |
| Linear Regression |    0.612551 |   0.575788 |
| Decision Tree     |    1.000000 |   0.622823 |
| Random Forest     |    0.973515 |   0.804624 |

## Final Model Comparison

### Classification

Logistic Regression achieved the highest test accuracy among the compared classification models, with a test accuracy of `0.964912`.

Random Forest achieved a test accuracy of `0.956140`, while Decision Tree achieved `0.912281`.

### Regression

Random Forest achieved the highest test R² score among the compared regression models, with a test R² score of `0.804624`.

Decision Tree achieved `0.622823`, while Linear Regression achieved `0.575788`.

## Overfitting Analysis

Decision Tree depth was varied using:

* `max_depth=2`
* `max_depth=4`
* `max_depth=6`
* `max_depth=10`
* `max_depth=None`

Train and test scores were compared to observe the effect of tree depth and identify overfitting.

## Feature Importance

Random Forest feature importance was calculated using `feature_importances_`.

The top 10 features were plotted for:

* Classification
* Regression

## Generated Outputs

* Decision Tree diagram
* Decision Tree depth vs score plot
* Random Forest classification feature importance plot
* Random Forest regression feature importance plot
* Saved model: `models/model.joblib`
* Metrics: `models/metrics.json`

## Testing

Day20 tests cover:

* Decision Tree depth analysis
* Regression depth analysis
* Random Forest feature importance
* Feature importance plot
* Decision Tree visualization
* Classification model comparison
* Regression model comparison
* Training CLI
* Metrics JSON
* Model training
* Model bundle saving
* CLI validation
* Saved model reload and prediction

Test result:

**29 passed**

Overall Epic4 coverage:

**80%**

`train.py` coverage:

**82%**
