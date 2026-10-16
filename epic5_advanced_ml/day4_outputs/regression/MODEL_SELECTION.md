# Model Selection and Validation Report

Task: regression
Selected model: random_forest

## Validation strategy

- Method: KFold
- Folds: 5
- Random state: 42
- Selection metric: neg_root_mean_squared_error

## Leakage checks

- The preprocessor is initialized using training data only.
- Preprocessing is fitted inside the cross-validation pipeline.
- The held-out test set is not used to rank candidate models.

## Hyperparameter Tuning

```json
{
    "random_forest": {
        "best_params": {
            "model__n_estimators": 100,
            "model__max_depth": null
        },
        "best_cv_score": -0.5173357666148459
    },
    "gradient_boosting": {
        "best_params": {
            "model__n_estimators": 150,
            "model__max_depth": 5,
            "model__learning_rate": 0.03
        },
        "best_cv_score": -0.5211210532251258
    }
}
```

## Leaderboard

            model  cv_mean   cv_std                                                                                          fold_scores training_time_seconds      mae      mse     rmse       r2
    random_forest 0.517336 0.011246  [0.5292926766244843, 0.5079401792054382, 0.508295487560729, 0.5327896520706118, 0.5083608376129664]                  None 0.331291 0.258856 0.508779 0.802461
gradient_boosting 0.521121 0.010958 [0.5321455699930371, 0.5165347428495369, 0.5110854932977151, 0.5361604459562844, 0.5096790140290554]                  None 0.358419 0.278324 0.527564 0.787606
         stacking 0.591682 0.010943  [0.6057217430677874, 0.582063257172179, 0.5762080962225845, 0.5992867305173418, 0.5951296938861053]              2.859734 0.393019 0.334151 0.578058 0.745003
              svm 0.603080 0.012336 [0.6050018132802636, 0.6010242044342604, 0.5836904896123046, 0.6224978584482671, 0.6031843721056352]             15.612904 0.408377 0.371619 0.609606 0.716410
              knn 0.662626 0.009065  [0.6663263755400558, 0.6673905734455268, 0.6484303329997915, 0.674340156842205, 0.6566449276971866]              0.178528 0.456818 0.451769 0.672138 0.655246
linear_regression 0.681183 0.022089 [0.6843974866036523, 0.6852085824677189, 0.6432319995897813, 0.6807092537935692, 0.7123696208569082]              0.100942 0.484711 0.456376 0.675556 0.651730
    decision_tree 0.752657 0.020228  [0.7782604224922414, 0.7197430234496759, 0.7666519985999505, 0.742653293300386, 0.7559784212780799]              0.412932 0.451060 0.497770 0.705528 0.620142

## Held-out test metrics

- **mae:** 0.331291
- **mse:** 0.258856
- **rmse:** 0.508779
- **r2:** 0.802461

## Cross-validation stability

- Mean: 0.517336
- Standard deviation: 0.011246
- Fold scores: [0.5292926766244843, 0.5079401792054382, 0.508295487560729, 0.5327896520706118, 0.5083608376129664]

A lower standard deviation indicates more consistent fold scores. Review the fold scores and the problem context before concluding that a model is stable.
