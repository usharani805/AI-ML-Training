# Model Selection

Task: regression

Selected model: random_forest

Selection is based on cross-validation performance.
Training time and model interpretability should also be considered when choosing a model for deployment.

## Leaderboard

            model  cv_mean   cv_std training_time_seconds      mae      mse     rmse       r2
    random_forest 0.517237 0.003619                  None 0.331291 0.258856 0.508779 0.802461
gradient_boosting 0.519753 0.006242                  None 0.358419 0.278324 0.527564 0.787606
         stacking 0.585983 0.008537              1.598614 0.393019 0.334151 0.578058 0.745003
              svm 0.602918 0.010623              9.063692 0.408377 0.371619 0.609606 0.716410
              knn 0.658935 0.013511              0.089207 0.456818 0.451769 0.672138 0.655246
linear_regression 0.681365 0.030302              0.058583 0.484711 0.456376 0.675556 0.651730
    decision_tree 0.742857 0.011924              0.236212 0.451060 0.497770 0.705528 0.620142

## Test Metrics

- **mae:** 0.331291
- **mse:** 0.258856
- **rmse:** 0.508779
- **r2:** 0.802461

## Cross-Validation

- Mean: 0.517237
- Standard deviation: 0.003619
