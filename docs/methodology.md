# Methodology

A short technical summary. The full discussion is in [`project_report.pdf`](project_report.pdf).

## 1. Problem

Predict a person's heart-disease risk band – **Low, Medium or High** – from 28 lifestyle and health features,
and explore natural groupings in the same population with unsupervised clustering.

## 2. Data preparation

| Step | What is done | Why |
|------|--------------|-----|
| Target | `pd.qcut(heart_disease_risk_score, q=3)` → Low / Medium / High | equal-frequency bands give balanced classes (≈ 6,487 per class in the test set) |
| Leakage removal | drop `diabetes_`, `hypertension_`, `obesity_`, `cholesterol_` and `heart_disease_risk_score` | these scores are derived from other columns; the hypertension score alone correlates r = 0.80 with the target |
| Encoding | one-hot encoding of categorical columns | 28 raw features become 40 model inputs |
| Split | stratified 80:20, `random_state=42` | 77,837 train / 19,460 test rows, same for every model |

Implementation: [`src/preprocessing.py`](../src/preprocessing.py).

## 3. Models

| Model | Configuration |
|-------|---------------|
| Decision Tree | `max_depth=10`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| XGBoost | `multi:softprob`; tuned by `RandomizedSearchCV` (20 candidates, 3-fold stratified CV, scoring = macro F1). Best: `n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.7, colsample_bytree=1.0, min_child_weight=3, gamma=0.5, reg_lambda=2` |

XGBoost needs integer labels, so Low / Medium / High are mapped to 0 / 1 / 2 for training and mapped back for display.

## 4. Evaluation

Weighted accuracy, precision, recall and F1 on the held-out test set, plus confusion matrices and per-class metrics.

| Model | Accuracy | F1 (wtd.) |
|-------|---------:|----------:|
| Decision Tree | 82.37 % | 82.42 % |
| Random Forest | 83.54 % | 83.62 % |
| XGBoost (tuned) | 85.01 % | 85.06 % |

Files: [`results/model_comparison.csv`](../results/model_comparison.csv),
[`results/per_class_metrics.csv`](../results/per_class_metrics.csv),
[`results/misclassifications.csv`](../results/misclassifications.csv).

Medium is the hardest class (F1 ≈ 74–78 %) because it sits between the other two bands; nearly all errors are to a neighbouring band.

## 5. WEKA clustering

`SimpleKMeans`, k = 3, Euclidean distance, seed 10, up to 500 iterations, on the 28 attributes normalised to 0–1.
The clusters hold 22,438 / 37,901 / 36,958 records (23 % / 39 % / 38 %) and are interpreted through their centroids. Plots: [`results/visualizations/weka_scatter.png`](../results/visualizations/weka_scatter.png).

## 6. Web app

`app/streamlit_app.py` shows the dataset, architecture, results and WEKA analysis, and serves the three trained
pipelines for a live prediction (single model or side-by-side comparison, with a what-if panel).
Models are trained by `python -m src.train_models <csv>` or from inside the app, and are saved in `models/` (not committed).
