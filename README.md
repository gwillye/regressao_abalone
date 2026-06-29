# Abalone Age Prediction (Linear Regression)

Academic project from UFMS (Data Science / Machine Learning). The idea is to predict an abalone's age from physical measurements using multiple linear regression, instead of the slow manual method of counting shell rings under a microscope. The repo includes the exploratory analysis, the visualizations and the model evaluation, all built with scikit-learn.

## Goal

Estimate abalone age (`Age = Rings + 1.5`) from 8 measurements that are easy to take, rather than counting rings by hand. The secondary question is whether a plain linear model is good enough to automate the prediction in the first place.

## Dataset

The data comes from the [Abalone Data Set on the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone). The files are included here (`abalone.data`, with the schema described in `abalone.names`).

- 4,177 instances, no missing values.
- Features: `Sex` (M/F/I, categorical), `Length`, `Diameter`, `Height`, `Whole weight`, `Shucked weight`, `Viscera weight`, `Shell weight`.
- Target: `Age` in years, derived from `Rings + 1.5`.

## How it works

The pipeline is split across a few scripts:

1. Initial exploration (`initial_exploration.py`): loads the data, renames columns, computes descriptive stats and checks types and missing values.
2. Visualization (`data_visualization.py`): produces histograms, an age-by-sex box plot and a correlation heatmap.
3. Modeling (`train_and_evaluate_model.py`): a scikit-learn `Pipeline` that one-hot encodes `Sex` and fits a `LinearRegression`, using an 80/20 train/test split with `random_state=42`.

## Results

The linear model lands at an R squared of 0.5482 and an RMSE of 2.21 years. In other words it explains roughly 55% of the variance in age, and the average error of about 2.2 years is reasonable given that ages span from 2.5 to 30.5 years. The most influential variables are `Shell weight` (r=0.63) and `Diameter` (r=0.57).

## Comparing several models

Beyond the linear baseline, `train_compare_models.py` compares a handful of regressors using 5-fold cross-validation on the full dataset. Running `python train_compare_models.py` writes the details to [`model_comparison_report.md`](model_comparison_report.md).

| Rank | Model | R² (CV) | RMSE | MAE |
|---|---|---|---|---|
| 1 | SVR (RBF) | 0.522 | 2.08 | 1.49 |
| 2 | Gradient Boosting | 0.466 | 2.16 | 1.57 |
| 3 | Random Forest | 0.446 | 2.19 | 1.60 |
| 4 | KNN (k=10) | 0.440 | 2.24 | 1.61 |
| 5 | Ridge / Linear | 0.419 | 2.24 | 1.64 |
| 6 | Decision Tree | -0.16 | 3.08 | 2.14 |

The best performer is SVR with an RBF kernel, which is about 25% better in R squared than the linear baseline. The non-linear models (SVR, gradient boosting) capture relationships the linear model misses. Even so, predicting exact age stays hard with an error around 2 years, so reframing the problem as age bands (a classification task) is a sensible next step.

## Roadmap

- Handle outliers (for example rows where `Height = 0`) and try some feature engineering.
- Explore the age-band classification framing and tune the SVR and gradient boosting hyper-parameters.

Reports: [`model_comparison_report.md`](model_comparison_report.md) and [`abalone_analysis_report.md`](abalone_analysis_report.md) (also available as PDF).

## Project structure

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

## How to run

```bash
# 1) environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) pipeline (scripts read abalone.data via relative path)
python initial_exploration.py
python data_visualization.py        # generates the .png files
python train_and_evaluate_model.py  # prints and saves R²/RMSE
```

## Stack

Python, pandas, scikit-learn, numpy, matplotlib, seaborn.

## Credit

Dataset: UCI ML Repository (Nash et al., 1994).
