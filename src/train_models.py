"""Train the Decision Tree, Random Forest and XGBoost models used by the web app.

Usage (from the repository root):
    python -m src.train_models data/health_risk_predictor.csv
    python -m src.train_models data/health_risk_predictor.csv --tune   # re-run the XGBoost search (slow)

This reproduces the models of the notebooks on the shared pipeline defined in
``src/preprocessing.py`` and saves, to ``models/``:
    dt_model.joblib, rf_model.joblib, xgb_model.joblib   (complete sklearn Pipelines)
    meta.json                                            (describes the app's input form)
    eval.json                                            (test-set metrics, confusion matrices, importances)

Models:
    Decision Tree : max_depth = 10
    Random Forest : n_estimators = 100
    XGBoost       : parameters found by RandomizedSearchCV in notebooks/xgboost.ipynb
                    (20 candidates, 3-fold stratified CV, scoring = macro F1)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import BINARY_COLS, NUM_TO_RISK, RISK, RISK_TO_NUM, load_and_split, make_preprocessor

ROOT = Path(__file__).resolve().parent.parent

# Best hyper-parameters found by the notebook's RandomizedSearchCV
XGB_BEST = dict(subsample=0.7, reg_lambda=2, n_estimators=500, min_child_weight=3,
                max_depth=4, learning_rate=0.1, gamma=0.5, colsample_bytree=1.0)
XGB_SEARCH_SPACE = {
    "classifier__n_estimators": [150, 250, 350, 500],
    "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "classifier__max_depth": [3, 4, 5, 6, 7],
    "classifier__min_child_weight": [1, 3, 5],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__gamma": [0, 0.1, 0.3, 0.5],
    "classifier__reg_lambda": [1, 2, 5, 10],
}


def evaluate(pipe, x_test, y_test_names, numeric_labels=False):
    pred = pipe.predict(x_test)
    if numeric_labels:
        pred = np.array([NUM_TO_RISK[int(v)] for v in pred])
    acc = accuracy_score(y_test_names, pred)
    p, r, f, _ = precision_recall_fscore_support(y_test_names, pred, average="weighted")
    cm = confusion_matrix(y_test_names, pred, labels=RISK)
    return acc, p, r, f, cm


def top_importances(pipe, k=15):
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    names = [n.split("__", 1)[1] if "__" in n else n for n in names]
    imp = pipe.named_steps["classifier"].feature_importances_
    order = np.argsort(imp)[::-1][:k]
    return [{"Feature": names[i], "Importance": float(imp[i])} for i in order]


def main(csv_path: str, tune: bool = False, out_dir: Path | None = None) -> None:
    out = Path(out_dir) if out_dir else ROOT / "models"
    out.mkdir(exist_ok=True)

    x, x_train, x_test, y_train, y_test, cat_cols, num_cols = load_and_split(csv_path)
    print(f"Loaded {len(x):,} rows x {x.shape[1]} features")
    print(f"Train {x_train.shape}, test {x_test.shape}")
    y_train_num = y_train.map(RISK_TO_NUM)

    pipes = {
        "dt_model": Pipeline([("preprocessor", make_preprocessor(cat_cols)),
                              ("classifier", DecisionTreeClassifier(random_state=42, max_depth=10))]),
        "rf_model": Pipeline([("preprocessor", make_preprocessor(cat_cols)),
                              ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]),
    }

    if tune:
        print("Running RandomizedSearchCV for XGBoost (this takes a while)...")
        search_base = Pipeline([("preprocessor", make_preprocessor(cat_cols)),
                                ("classifier", XGBClassifier(objective="multi:softprob", num_class=3,
                                                             eval_metric="mlogloss", random_state=42, n_jobs=1))])
        search = RandomizedSearchCV(search_base, XGB_SEARCH_SPACE, n_iter=20, scoring="f1_macro",
                                    cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                    random_state=42, n_jobs=-1, verbose=1, refit=True)
        search.fit(x_train, y_train_num)
        print("Best params:", search.best_params_, "| CV macro F1:", round(search.best_score_, 4))
        xgb_pipe = search.best_estimator_
    else:
        xgb_pipe = Pipeline([("preprocessor", make_preprocessor(cat_cols)),
                             ("classifier", XGBClassifier(objective="multi:softprob", num_class=3,
                                                          eval_metric="mlogloss", random_state=42,
                                                          n_jobs=-1, **XGB_BEST))])
        xgb_pipe.fit(x_train, y_train_num)
    pipes["xgb_model"] = xgb_pipe

    display = {"dt_model": "Decision Tree", "rf_model": "Random Forest", "xgb_model": "XGBoost (tuned)"}
    summary = {"n_train": int(len(x_train)), "n_test": int(len(x_test)), "models": {}}
    for name, model in pipes.items():
        if name != "xgb_model":
            model.fit(x_train, y_train)
        acc, p, r, f, cm = evaluate(model, x_test, y_test, numeric_labels=(name == "xgb_model"))
        print(f"{display[name]:16s} accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")
        joblib.dump(model, out / f"{name}.joblib", compress=3)
        summary["models"][display[name]] = {
            "accuracy": acc, "precision": p, "recall": r, "f1": f,
            "confusion": cm.tolist(),
            "importance": top_importances(model) if name != "dt_model" else [],
        }
    (out / "eval.json").write_text(json.dumps(summary, indent=2))

    # Metadata used by the app to build the prediction form from the real data
    meta = {
        "feature_order": x.columns.tolist(),
        "categorical": {c: sorted(x[c].dropna().unique().tolist()) for c in cat_cols},
        "binary": BINARY_COLS,
        "numeric": {
            c: {
                "min": float(x[c].min()),
                "max": float(x[c].max()),
                "median": float(x[c].median()),
                "is_int": bool(pd.api.types.is_integer_dtype(x[c])),
            }
            for c in num_cols
            if c not in BINARY_COLS
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models, meta.json and eval.json to {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("Usage: python -m src.train_models path/to/health_risk_predictor.csv [--tune]")
    main(args[0], tune="--tune" in sys.argv)
