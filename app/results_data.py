"""
Verified project results used by the presentation app.

Everything here comes from the team's notebooks / the project report:
  * Confusion matrices are the 19,460-record test-set counts (rows = actual,
    columns = predicted, order Low, Medium, High; row sums are 6487 / 6486 / 6487).
  * All accuracy / precision / recall / F1 values are COMPUTED from those counts,
    so the tables and charts can never disagree with the matrices.
  * XGBoost numbers are for the tuned model (RandomizedSearchCV, 20 candidates,
    3-fold stratified CV, macro-F1 scoring).
"""
import numpy as np
import pandas as pd

RISK = ["Low", "Medium", "High"]
MODELS = ["Decision Tree", "Random Forest", "XGBoost (tuned)"]
OWNERS = {"Decision Tree": "Adithya Dev", "Random Forest": "Midhun M. Pillai",
          "XGBoost (tuned)": "Joyal John Mathew"}

CONFUSION = {
    "Decision Tree": [[5629, 856, 2], [932, 4836, 718], [6, 916, 5565]],
    "Random Forest": [[5593, 894, 0], [828, 5006, 652], [2, 827, 5658]],
    "XGBoost (tuned)": [[5666, 821, 0], [779, 5105, 602], [2, 713, 5772]],
}


def metrics_from_cm(cm):
    cm = np.array(cm, dtype=float)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    precision = tp / cm.sum(axis=0)
    recall = tp / support
    f1 = 2 * precision * recall / (precision + recall)
    w = support / support.sum()
    return {
        "per_class": pd.DataFrame({"Class": RISK, "Precision": precision, "Recall": recall,
                                   "F1-score": f1, "Support": support.astype(int)}),
        "accuracy": tp.sum() / support.sum(),
        "precision": (w * precision).sum(),
        "recall": (w * recall).sum(),
        "f1": (w * f1).sum(),
    }


METRICS = {m: metrics_from_cm(CONFUSION[m]) for m in MODELS}


def comparison_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        r = METRICS[m]
        rows.append({"Model": m, "Accuracy": r["accuracy"] * 100, "Precision (wtd.)": r["precision"] * 100,
                     "Recall (wtd.)": r["recall"] * 100, "F1-score (wtd.)": r["f1"] * 100})
    return pd.DataFrame(rows)


def error_table() -> pd.DataFrame:
    """Count of each kind of misclassification for every model."""
    rows = []
    for m in MODELS:
        cm = np.array(CONFUSION[m])
        for i, a in enumerate(RISK):
            for j, p in enumerate(RISK):
                if i != j:
                    rows.append({"Model": m, "Error": f"{a} → {p}", "Count": int(cm[i, j]),
                                 "Kind": "Adjacent bands" if abs(i - j) == 1 else "Low ↔ High"})
    return pd.DataFrame(rows)


ERROR_ORDER = ["Low → Medium", "Medium → Low", "Medium → High", "High → Medium",
               "Low → High", "High → Low"]

# XGBoost feature importance - baseline model (Colab notebook, Step 7.6)
XGB_IMPORTANCE = pd.DataFrame(
    [("Age", 0.276659), ("smoking_status_Former", 0.171062), ("smoking_status_Never", 0.101799),
     ("systolic_bp", 0.064223), ("hba1c", 0.049445), ("ldl_cholesterol", 0.047356),
     ("hdl_cholesterol", 0.033101), ("family_history_diabetes", 0.022304),
     ("cholesterol_total", 0.019569), ("glucose_postprandial", 0.013529),
     ("glucose_fasting", 0.012704), ("bmi", 0.009508), ("waist_to_hip_ratio", 0.007574),
     ("physical_activity_minutes_per_week", 0.007407), ("income_level_Low", 0.006837)],
    columns=["Feature", "Importance"])

XGB_PARAMS = pd.DataFrame(
    [["n_estimators", "150, 250, 350, 500", "200", "500"],
     ["learning_rate", "0.03, 0.05, 0.08, 0.1", "0.1", "0.1"],
     ["max_depth", "3, 4, 5, 6, 7", "6", "4"],
     ["min_child_weight", "1, 3, 5", "1 (default)", "3"],
     ["subsample", "0.7, 0.8, 0.9, 1.0", "0.8", "0.7"],
     ["colsample_bytree", "0.7, 0.8, 0.9, 1.0", "0.8", "1.0"],
     ["gamma", "0, 0.1, 0.3, 0.5", "0 (default)", "0.5"],
     ["reg_lambda", "1, 2, 5, 10", "1 (default)", "2"]],
    columns=["Hyperparameter", "Search space", "Baseline", "Selected"])

XGB_BASE_VS_TUNED = pd.DataFrame(
    [["Baseline XGBoost", 84.98, 0.8503], ["Tuned XGBoost", 85.01, 0.8505]],
    columns=["Model", "Accuracy (%)", "Macro F1"])

WEKA_PARAMS = pd.DataFrame(
    [["Tool / algorithm", "WEKA SimpleKMeans"], ["Number of clusters", "3"],
     ["Distance function", "Euclidean"], ["Initialisation", "Random"],
     ["Maximum iterations", "500"], ["Seed", "10"], ["Evaluation", "Training set"],
     ["Attributes used", "28 (normalised 0-1)"]],
    columns=["Parameter", "Value"])
WEKA_DIST = pd.DataFrame({"Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
                          "Records": [22438, 37901, 36958], "Percentage": [23, 39, 38]})
WEKA_CENTROIDS = pd.DataFrame(
    [["Age", 0.4397, 0.4379, 0.4406], ["BMI", 0.4384, 0.4390, 0.4384],
     ["Family history of diabetes", 0.3399, 0.2596, 0.1057],
     ["Fasting glucose", 0.4681, 0.4598, 0.4459],
     ["Postprandial glucose", 0.4242, 0.4166, 0.4073],
     ["HbA1c", 0.4448, 0.4368, 0.4260], ["Systolic BP", 0.2892, 0.2899, 0.2903]],
    columns=["Feature", "Cluster 0", "Cluster 1", "Cluster 2"])

# Hypothetical person used in the notebook / report demonstration
SAMPLE_PERSON = {
    "Age": 45, "gender": "Male", "ethnicity": "Asian", "education_level": "Graduate",
    "income_level": "Middle", "employment_status": "Employed", "smoking_status": "Never",
    "alcohol_consumption_per_week": 1, "physical_activity_minutes_per_week": 150,
    "diet_score": 6.5, "sleep_hours_per_day": 7.0, "screen_time_hours_per_day": 5.0,
    "family_history_diabetes": 0, "hypertension_history": 0, "cardiovascular_history": 0,
    "bmi": 24.5, "waist_to_hip_ratio": 0.88, "systolic_bp": 125, "diastolic_bp": 80,
    "heart_rate": 72, "cholesterol_total": 190, "hdl_cholesterol": 50, "ldl_cholesterol": 115,
    "triglycerides": 140, "glucose_fasting": 95, "glucose_postprandial": 140,
    "insulin_level": 8.0, "hba1c": 5.6,
}
# What the notebook printed for that person (reported, tuned XGBoost / Random Forest)
SAMPLE_REPORTED = {"XGBoost (tuned)": {"Low": 55.92, "Medium": 43.91, "High": 0.17},
                   "Random Forest": {"Low": 49.0, "Medium": 48.0, "High": 3.0}}

LEAKAGE_COLS = ["diabetes_risk_score", "hypertension_risk_score", "obesity_risk_score",
                "cholesterol_risk_score", "heart_disease_risk_score"]
