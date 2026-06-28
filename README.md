# 🐚 Abalone Age Prediction — Linear Regression

> *Academic context — UFMS, Data Science / Machine Learning.*

A **machine-learning** project (multiple linear regression) that predicts an abalone's age from physical measurements — replacing the slow manual method of counting shell rings under a microscope. Includes exploratory analysis, visualizations and model evaluation with `scikit-learn`.

## 🎯 Goal
Estimate abalone age (`Age = Rings + 1.5`) from 8 easy-to-take physical measurements instead of counting rings, and assess whether a linear model is viable for automating the prediction.

## 📊 Dataset
[Abalone Data Set — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone) (included: `abalone.data`, schema in `abalone.names`).
- **4,177 instances**, no missing values.
- **Features:** `Sex` (M/F/I — categorical), `Length`, `Diameter`, `Height`, `Whole weight`, `Shucked weight`, `Viscera weight`, `Shell weight`.
- **Target:** `Age` (years), derived from `Rings + 1.5`.

## 🧪 Methodology
1. **Initial exploration** (`initial_exploration.py`): load, column renaming, descriptive stats, type/missing checks.
2. **Visualization** (`data_visualization.py`): histograms, age-by-sex box plot, correlation heatmap.
3. **Modeling** (`train_and_evaluate_model.py`): scikit-learn `Pipeline` with `OneHotEncoder` for `Sex` + `LinearRegression`; 80/20 train/test split (`random_state=42`).

## 📈 Results
| Metric | Value |
|---|---|
| **R²** (coefficient of determination) | **0.5482** |
| **RMSE** (mean error) | **2.21 years** |

The model explains ~55% of age variance; the ~2.2-year mean error is reasonable given the 2.5–30.5-year range. The most influential variables are `Shell weight` (r=0.63) and `Diameter` (r=0.57).

## 🏆 Model comparison (`train_compare_models.py`)
Beyond the linear baseline, several regressors were compared with **5-fold cross-validation** (full dataset). Run it with `python train_compare_models.py` → writes [`model_comparison_report.md`](model_comparison_report.md).

| Rank | Model | R² (CV) | RMSE | MAE |
|---|---|---|---|---|
| 1 | **SVR (RBF)** ⭐ | **0.522** | 2.08 | 1.49 |
| 2 | Gradient Boosting | 0.466 | 2.16 | 1.57 |
| 3 | Random Forest | 0.446 | 2.19 | 1.60 |
| 4 | KNN (k=10) | 0.440 | 2.24 | 1.61 |
| 5 | Ridge / Linear | 0.419 | 2.24 | 1.64 |
| 6 | Decision Tree | −0.16 | 3.08 | 2.14 |

**Best: SVR (RBF) — ~+25% R² over the linear baseline.** Non-linear models (SVR, gradient boosting) capture relationships the linear model misses; predicting exact age stays hard (~2-year error), so framing it as **age bands (classification)** is a sensible next step.

## 📌 Roadmap
- Handle **outliers** (e.g. `Height = 0`) and engineer features.
- Try the **age-band classification** framing and tune the SVR/boosting hyper-parameters.

📄 Reports: [`model_comparison_report.md`](model_comparison_report.md) · [`abalone_analysis_report.md`](abalone_analysis_report.md) (also PDF).

## 🗂️ Structure
```
.
├── initial_exploration.py        # EDA: load + descriptive stats
├── data_visualization.py         # generates the 3 charts (.png)
├── train_and_evaluate_model.py   # regression pipeline + metrics
├── abalone.data / abalone.names  # UCI dataset + schema
├── histograms.png                # variable distributions
├── age_by_sex_boxplot.png        # age by sex
├── correlation_heatmap.png       # variable correlations
├── abalone_analysis_report.md/.pdf
├── model_evaluation_results.txt  # R² and RMSE saved by the run
├── requirements.txt
└── .gitignore
```

## ▶️ How to run
```bash
# 1) environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) pipeline (scripts read abalone.data via relative path)
python initial_exploration.py
python data_visualization.py        # generates the .png files
python train_and_evaluate_model.py  # prints and saves R²/RMSE
```

## 🛠️ Stack
Python · pandas · scikit-learn · numpy · matplotlib · seaborn

## 📜 Credit
Dataset: UCI ML Repository (Nash et al., 1994).
