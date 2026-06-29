# Abalone, Model Comparison Report

These are the results of a 5-fold cross-validation run on the full dataset, ranked by R² (the higher the better; RMSE and MAE are in years, where lower is better).

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | SVR (RBF) | 0.5217 | 2.0835 | 1.4931 |
| 2 | Gradient Boosting | 0.4656 | 2.1616 | 1.5686 |
| 3 | Random Forest | 0.4463 | 2.1938 | 1.5955 |
| 4 | KNN (k=10) | 0.4396 | 2.2356 | 1.6146 |
| 5 | Ridge | 0.4194 | 2.2409 | 1.6394 |
| 6 | Linear Regression | 0.4186 | 2.2417 | 1.6394 |
| 7 | Decision Tree | -0.1607 | 3.0836 | 2.1426 |

## Best model: SVR (RBF)

- Cross-validated R² = 0.5217, against a Linear Regression baseline of 0.4186, which is a +24.6% gain in R².
- On the held-out test split (80/20): R² = 0.5647, RMSE = 2.1707 yrs, MAE = 1.4852 yrs.

## Takeaways

- The tree ensembles (Random Forest and Gradient Boosting) pick up the non-linear relationships that the linear baseline simply misses.
- Predicting the exact age is still hard, with an error of roughly 2 years. That lines up with the dataset's known ceiling, so reframing the problem as age bands (a classification task) is a sensible next step.
