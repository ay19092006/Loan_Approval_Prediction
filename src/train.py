import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src import config
from src.preprocessing import build_preprocessor

def get_models() -> dict[str, Pipeline]:
    rs = config.RANDOM_STATE
    return {
        "Baseline (majority class)": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=rs)),
        ]),
        "Decision Tree": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=rs)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                           min_samples_leaf=5, n_jobs=-1, random_state=rs)),
        ]),
    }

def cross_validate(models: dict, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    out = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        out[name] = {"cv_f1_mean": float(np.mean(scores)), "cv_f1_std": float(np.std(scores))}
        print(f"[cv] {name:28s} F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return out

def fit_all(models: dict, X_train, y_train) -> dict:
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models