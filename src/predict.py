import joblib
import pandas as pd
from src import config
from src.preprocessing import add_features, feature_columns

def load_model():
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{config.MODEL_FILE} not found. Run `python main.py` first to train the model."
        )
    return joblib.load(config.MODEL_FILE)

def predict_one(model, applicant: dict) -> tuple[str, float]:
    num, cat = feature_columns()
    row = add_features(pd.DataFrame([applicant]))[num + cat]
    p = float(model.predict_proba(row)[0, 1])
    return (config.POSITIVE_LABEL if p >= 0.5 else config.NEGATIVE_LABEL), p