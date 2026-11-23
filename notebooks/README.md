# Notebooks

Each notebook is **standalone**: it loads the dataset, prepares it the same way, trains one algorithm and evaluates it.
Run them in Google Colab (*File → Upload notebook*) and upload `health_risk_predictor.csv` when asked.

| Notebook | Algorithm | Status |
|----------|-----------|--------|
| `01_decision_tree.ipynb` | Decision Tree (`max_depth=10`) | done |
| `02_random_forest.ipynb` | Random Forest (100 trees) | done |
| `03_xgboost.ipynb` | XGBoost with `RandomizedSearchCV` tuning | done |
| `04_weka_clustering` | WEKA SimpleKMeans, k = 3 (seed 10, 500 iterations) | done in the WEKA Explorer |

## Common preparation (identical in every notebook)

1. Create the target: `pd.qcut(heart_disease_risk_score, 3, labels=["Low", "Medium", "High"])`
2. Drop the five leakage columns (`*_risk_score`)
3. One-hot encode categorical columns (28 → 40 features)
4. Stratified 80:20 split with `random_state=42`

The same steps live in [`../src/preprocessing.py`](../src/preprocessing.py) for scripts and the web app.

## Note on numbers

The notebooks use `pd.get_dummies(drop_first=True)`; the web app uses a scikit-learn `OneHotEncoder` pipeline.
The two encodings are equivalent for tree models, but results can differ by a fraction of a percent between
library versions. The figures in the report and README come from the original Colab runs.
