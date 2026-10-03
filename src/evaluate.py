import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from src import config
from src.preprocessing import feature_columns

LABELS = [config.NEGATIVE_LABEL, config.POSITIVE_LABEL]

def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=160)
    plt.close(fig)
    return name

def evaluate_models(models: dict, X_test, y_test) -> dict:
    results, probas = {}, {}
    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        probas[name] = proba
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        r = results[name]
        print(f"[test] {name:28s} acc={r['accuracy']:.3f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")

    figs = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        sns.heatmap(np.array(r["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    figs.append(_save(fig, "07_confusion_matrices.png"))
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(fontsize=8, loc="lower right")
    figs.append(_save(fig, "08_roc_curves.png"))

    df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.set_ylabel("Score")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="lower right", ncol=4, fontsize=8)
    figs.append(_save(fig, "09_model_comparison.png"))
    return {"metrics": results, "figures": figs}

def feature_importance(model, top_n: int = 10) -> dict:
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
    top = imp.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set_title("Random Forest - top feature importances")
    ax.set_xlabel("Importance")
    name = _save(fig, "10_feature_importance.png")
    return {"figure": name, "top": {k: round(float(v), 4) for k, v in imp.head(top_n).items()}}