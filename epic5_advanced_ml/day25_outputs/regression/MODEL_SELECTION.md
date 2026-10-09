# Model Selection — Regression

## Selected Model: Random Forest Regressor

* **Cross-validation RMSE:** 0.5237
* **Cross-validation standard deviation:** 0.0041
* **Test RMSE:** 0.5147
* **Test MAE:** 0.3355
* **Test R²:** 0.7979
* **Training time:** 6.923 seconds

## Performance Comparison

Random Forest achieved the lowest cross-validation RMSE among the candidate models. Gradient Boosting had a cross-validation RMSE of 0.5312 and took 14.887 seconds to train.

## Training Time and Interpretability

* **Random Forest:** Strong predictive performance, but its multiple trees make it harder to interpret than a simple linear model.
* **Gradient Boosting:** Achieved competitive results but required more training time in this run.
* **Linear Regression:** Easier to interpret, but its cross-validation RMSE was higher at 0.6836.

## Final Decision

Random Forest was selected because it achieved the lowest cross-validation RMSE and strong test performance. Lower RMSE indicates better predictive accuracy. Model choice should also consider interpretability, training time, and the intended use case.
