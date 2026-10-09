# Model Selection — Classification

## Selected Model: Logistic Regression

* **Cross-validation F1 mean:** 0.9845
* **Cross-validation standard deviation:** 0.0083
* **Test accuracy:** 0.9825
* **Test F1 score:** 0.9861
* **Test ROC-AUC:** 0.9954
* **Training time:** 7.170 seconds

## Performance Comparison

Logistic Regression achieved the highest cross-validation F1 score in the candidate-model leaderboard. Stacking achieved a similar F1 score and trained faster, taking 0.488 seconds.

## Training Time and Interpretability

* **Logistic Regression:** Faster to interpret because its coefficients show how features relate to predictions. Training took longer than stacking in this run.
* **Stacking:** Combines multiple models and achieved similar test performance, but is more difficult to interpret.
* **Decision Tree:** Easier to visualize and explain, but had lower F1 performance in this comparison.

## Final Decision

Logistic Regression was selected because it achieved the best cross-validation F1 score while maintaining strong test accuracy and ROC-AUC. Results may vary with different data and settings.
