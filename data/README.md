# Data

The dataset is **not committed** to the repository (it is large and regenerable). Place it here as:

```
data/health_risk_predictor.csv
```

## Description

| Item | Value |
|------|-------|
| Rows | 97,297 |
| Columns | 33 (28 input features + 5 pre-computed risk scores) |
| Target | `heart_disease_risk_score`, converted to **Low / Medium / High** by equal-frequency (quantile) binning |

## Leakage columns (removed before training)

`diabetes_risk_score`, `hypertension_risk_score`, `obesity_risk_score`, `cholesterol_risk_score` and `heart_disease_risk_score`
are derived from the other columns, so a model could "read the answer" from them. They are dropped in
[`src/preprocessing.py`](../src/preprocessing.py).

## Split used everywhere

Stratified 80:20 split with `random_state=42` → 77,837 training rows and 19,460 test rows.
Every model must use this split (call `src.preprocessing.load_and_split`) so results stay comparable.

The Streamlit app's **Live Risk Predictor** page can also take the CSV as an upload and train the models in the browser.
