"""Load the trained models and turn a person's measurements into a risk prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import RISK

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_FILES = {
    "Decision Tree": "dt_model.joblib",
    "Random Forest": "rf_model.joblib",
    "XGBoost (tuned)": "xgb_model.joblib",
}


def load_models(models_dir: Path = MODELS_DIR) -> dict:
    """Load every saved model pipeline that exists in ``models_dir``."""
    return {
        label: joblib.load(models_dir / fname)
        for label, fname in MODEL_FILES.items()
        if (models_dir / fname).exists()
    }


def class_probs(model, row: pd.DataFrame) -> dict:
    """Probability of Low / Medium / High for a one-row DataFrame.

    The XGBoost model was trained on integer labels (0, 1, 2), the other two on
    strings, so the class names are mapped back here.
    """
    p = model.predict_proba(row)[0]
    names = [c if isinstance(c, str) else RISK[int(c)] for c in model.classes_]
    d = dict(zip(names, p))
    return {k: float(d.get(k, 0.0)) for k in RISK}


def predict_risk(model, values: dict, feature_order: list[str]) -> tuple[str, dict]:
    """Predict the risk class for one person.

    ``values`` maps each raw feature name to its value (binary history columns as
    0/1). Returns the predicted class and the class probabilities.
    """
    row = pd.DataFrame([values])[feature_order]
    probs = class_probs(model, row)
    return max(probs, key=probs.get), probs
