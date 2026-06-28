# Abalone — Model Comparison Report

5-fold cross-validation on the full dataset, ranked by R² (higher is better; RMSE/MAE in years, lower is better).

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | SVR (RBF) ⭐ | 0.5217 | 2.0835 | 1.4931 |
| 2 | Gradient Boosting | 0.4656 | 2.1616 | 1.5686 |
| 3 | Random Forest | 0.4463 | 2.1938 | 1.5955 |
| 4 | KNN (k=10) | 0.4396 | 2.2356 | 1.6146 |
| 5 | Ridge | 0.4194 | 2.2409 | 1.6394 |
| 6 | Linear Regression | 0.4186 | 2.2417 | 1.6394 |
| 7 | Decision Tree | -0.1607 | 3.0836 | 2.1426 |

## Best model: **SVR (RBF)**
- Cross-validated R² = **0.5217** (vs Linear Regression baseline 0.4186 → **+24.6%** R²).
- Held-out test (80/20): R²=0.5647, RMSE=2.1707 yrs, MAE=1.4852 yrs.

## Takeaways
- Tree ensembles (Random Forest / Gradient Boosting) capture the non-linear relationships the linear baseline misses.
- Predicting exact age stays hard (~2-year error) — consistent with the dataset's known ceiling; framing it as age *bands* (classification) is a sensible next step.
