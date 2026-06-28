"""Abalone age prediction — multi-model comparison.

Extends the original linear-regression baseline: trains several regressors,
compares them with 5-fold cross-validation (R2, RMSE, MAE), evaluates the best
on a held-out test set, and writes a Markdown report ranking the models.

Run:  python train_compare_models.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

COLS = ["Sex", "Length", "Diameter", "Height", "Whole weight",
        "Shucked weight", "Viscera weight", "Shell weight", "Rings"]


def load():
    df = pd.read_csv("abalone.data", header=None, names=COLS)
    df["Age"] = df["Rings"] + 1.5
    df = df.drop(columns=["Rings"])
    return df


def make_pipeline(model, scale_numeric: bool):
    num = ["Length", "Diameter", "Height", "Whole weight",
           "Shucked weight", "Viscera weight", "Shell weight"]
    cat = ["Sex"]
    num_tf = StandardScaler() if scale_numeric else "passthrough"
    pre = ColumnTransformer([("num", num_tf, num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = load()
    X = df.drop(columns=["Age"])
    y = df["Age"]

    models = {
        "Linear Regression":   (LinearRegression(), True),
        "Ridge":               (Ridge(alpha=1.0), True),
        "Decision Tree":       (DecisionTreeRegressor(random_state=42), False),
        "Random Forest":       (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
        "Gradient Boosting":   (GradientBoostingRegressor(random_state=42), False),
        "SVR (RBF)":           (SVR(C=10, gamma="scale"), True),
        "KNN (k=10)":          (KNeighborsRegressor(n_neighbors=10), True),
    }

    scoring = {"r2": "r2",
               "rmse": "neg_root_mean_squared_error",
               "mae": "neg_mean_absolute_error"}

    rows = []
    print("5-fold cross-validation (full dataset):\n")
    for name, (est, scale) in models.items():
        pipe = make_pipeline(est, scale)
        cv = cross_validate(pipe, X, y, cv=5, scoring=scoring, n_jobs=-1)
        r2 = cv["test_r2"].mean()
        rmse = -cv["test_rmse"].mean()
        mae = -cv["test_mae"].mean()
        rows.append((name, r2, rmse, mae))
        print(f"  {name:<20} R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    rows.sort(key=lambda r: r[1], reverse=True)  # rank by R2
    best_name = rows[0][0]
    best_est, best_scale = models[best_name]

    # final held-out evaluation of the best model
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    best_pipe = make_pipeline(best_est, best_scale).fit(X_tr, y_tr)
    pred = best_pipe.predict(X_te)
    hold_r2 = r2_score(y_te, pred)
    hold_rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    hold_mae = mean_absolute_error(y_te, pred)

    base_r2 = next(r for n, r, *_ in rows if n == "Linear Regression")
    lift = (rows[0][1] - base_r2) / base_r2 * 100

    # write the report
    lines = ["# Abalone — Model Comparison Report", "",
             "5-fold cross-validation on the full dataset, ranked by R² "
             "(higher is better; RMSE/MAE in years, lower is better).", "",
             "| Rank | Model | R² | RMSE | MAE |", "|---|---|---|---|---|"]
    for i, (name, r2, rmse, mae) in enumerate(rows, 1):
        mark = " ⭐" if name == best_name else ""
        lines.append(f"| {i} | {name}{mark} | {r2:.4f} | {rmse:.4f} | {mae:.4f} |")
    lines += ["",
              f"## Best model: **{best_name}**",
              f"- Cross-validated R² = **{rows[0][1]:.4f}** "
              f"(vs Linear Regression baseline {base_r2:.4f} → **{lift:+.1f}%** R²).",
              f"- Held-out test (80/20): R²={hold_r2:.4f}, RMSE={hold_rmse:.4f} yrs, MAE={hold_mae:.4f} yrs.",
              "",
              "## Takeaways",
              "- Tree ensembles (Random Forest / Gradient Boosting) capture the "
              "non-linear relationships the linear baseline misses.",
              "- Predicting exact age stays hard (~2-year error) — consistent with "
              "the dataset's known ceiling; framing it as age *bands* (classification) "
              "is a sensible next step.",
              ""]
    with open("model_comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nBest: {best_name} (CV R²={rows[0][1]:.4f}, {lift:+.1f}% vs linear)")
    print("Report written: model_comparison_report.md")


if __name__ == "__main__":
    main()
