# Model Selection

Task: classification

Selected model: logistic_regression

Selection is based on cross-validation performance.
Training time and model interpretability should also be considered when choosing a model for deployment.

## Leaderboard

              model  cv_mean   cv_std training_time_seconds  accuracy  precision   recall       f1  roc_auc
logistic_regression 0.984525 0.012517                  None  0.973684   0.972603 0.986111 0.979310 0.995701
           stacking 0.981018 0.019954                  None  0.982456   0.972973 1.000000 0.986301 0.991071
                svm 0.977292 0.014038              0.037239  0.982456   0.986111 0.986111 0.986111 0.995040
                knn 0.973999 0.016362               0.00875  0.956140   0.958904 0.972222 0.965517 0.978836
  gradient_boosting 0.965030 0.010638              0.460513  0.956140   0.946667 0.986111 0.965986 0.990741
      random_forest 0.963408 0.018366              0.217389  0.956140   0.958904 0.972222 0.965517 0.993882
      decision_tree 0.928176 0.014981                0.0311  0.912281   0.955882 0.902778 0.928571 0.915675

## Test Metrics

- **accuracy:** 0.973684
- **precision:** 0.972603
- **recall:** 0.986111
- **f1:** 0.979310
- **roc_auc:** 0.995701

## Cross-Validation

- Mean: 0.984525
- Standard deviation: 0.012517
