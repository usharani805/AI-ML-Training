# Model Selection and Validation Report

Task: classification
Selected model: logistic_regression

## Validation strategy

- Method: StratifiedKFold
- Folds: 5
- Random state: 42
- Selection metric: f1

## Leakage checks

- The preprocessor is initialized using training data only.
- Preprocessing is fitted inside the cross-validation pipeline.
- The held-out test set is not used to rank candidate models.

## Hyperparameter Tuning

```json
{
    "logistic_regression": {
        "best_params": {
            "model__C": 0.1
        },
        "best_cv_score": 0.9861164154764722
    },
    "svm": {
        "best_params": {
            "model__kernel": "linear",
            "model__C": 0.1
        },
        "best_cv_score": 0.9810153888265501
    }
}
```

## Leaderboard

              model  cv_mean   cv_std                                                                                          fold_scores training_time_seconds  accuracy  precision   recall       f1  roc_auc
logistic_regression 0.986116 0.004237   [0.9827586206896551, 0.9824561403508771, 0.991304347826087, 0.9827586206896551, 0.991304347826087]                  None  0.973684   0.972603 0.986111 0.979310 0.995701
                svm 0.981015 0.008191  [0.9661016949152542, 0.9824561403508771, 0.9824561403508771, 0.9827586206896551, 0.991304347826087]                  None  0.982456   0.986111 0.986111 0.986111 0.993717
           stacking 0.975368 0.010492 [0.9827586206896551, 0.9911504424778761, 0.9642857142857143, 0.9743589743589743, 0.9642857142857143]              0.194817  0.982456   0.972973 1.000000 0.986301 0.991071
                knn 0.970489 0.009111 [0.9743589743589743, 0.9743589743589743, 0.9557522123893806, 0.9655172413793104, 0.9824561403508771]              0.040606  0.956140   0.958904 0.972222 0.965517 0.978836
      random_forest 0.969935 0.014605 [0.9734513274336283, 0.9642857142857143, 0.9464285714285714, 0.9743589743589743, 0.9911504424778761]              0.362425  0.956140   0.958904 0.972222 0.965517 0.993882
  gradient_boosting 0.961363 0.012073 [0.9649122807017544, 0.9557522123893806, 0.9464285714285714, 0.9572649572649573, 0.9824561403508771]              0.625107  0.956140   0.946667 0.986111 0.965986 0.990741
      decision_tree 0.931901 0.014435 [0.9464285714285714, 0.9464285714285714, 0.9174311926605505, 0.9122807017543859, 0.9369369369369369]              0.039344  0.912281   0.955882 0.902778 0.928571 0.915675

## Held-out test metrics

- **accuracy:** 0.973684
- **precision:** 0.972603
- **recall:** 0.986111
- **f1:** 0.979310
- **roc_auc:** 0.995701

## Cross-validation stability

- Mean: 0.986116
- Standard deviation: 0.004237
- Fold scores: [0.9827586206896551, 0.9824561403508771, 0.991304347826087, 0.9827586206896551, 0.991304347826087]

A lower standard deviation indicates more consistent fold scores. Review the fold scores and the problem context before concluding that a model is stable.
