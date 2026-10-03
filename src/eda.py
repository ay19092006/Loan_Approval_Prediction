import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import config

PALETTE = {"Approved": "#2a7f62", "Rejected": "#c8553d"}
sns.set_theme(style="whitegrid", font_scale=1.0)

def _save(fig, name: str) -> str:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return name

def run_eda(df: pd.DataFrame) -> dict:
    saved = []
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=PALETTE, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Target class distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Applications")
    saved.append(_save(fig, "01_class_distribution.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(data=df, x="cibil_score", hue=config.TARGET, palette=PALETTE,
                 bins=30, element="step", ax=ax)
    ax.set_title("Credit (CIBIL) score by loan outcome")
    ax.set_xlabel("CIBIL score")
    saved.append(_save(fig, "02_cibil_by_status.png"))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    num_cols = config.NUMERIC_FEATURES.copy()
    tmp = df[num_cols].copy()
    tmp["approved"] = (df[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    sns.heatmap(tmp.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix (numeric features + approval)")
    saved.append(_save(fig, "03_correlation_heatmap.png"))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ratio = (df["loan_amount"] / df["income_annum"]).rename("loan_to_income")
    sns.boxplot(x=df[config.TARGET], y=ratio, hue=df[config.TARGET],
                palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Loan amount relative to annual income")
    ax.set_xlabel("")
    ax.set_ylabel("Loan / income")
    saved.append(_save(fig, "04_loan_to_income_box.png"))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, col in zip(axes, ["education", "self_employed", "no_of_dependents"]):
        rate = (df.assign(ok=(df[config.TARGET] == config.POSITIVE_LABEL))
                  .groupby(col)["ok"].mean().mul(100))
        sns.barplot(x=rate.index.astype(str), y=rate.values, color="#3b6ea5", ax=ax)
        ax.set_ylim(0, 100)
        ax.set_title(f"Approval rate by {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Approved (%)")
    saved.append(_save(fig, "05_approval_rate_categorical.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    rate = (df.assign(ok=(df[config.TARGET] == config.POSITIVE_LABEL))
              .groupby("loan_term")["ok"].mean().mul(100))
    ax.plot(rate.index, rate.values, marker="o", color="#3b6ea5")
    ax.set_ylim(0, 100)
    ax.set_title("Approval rate by loan term (years)")
    ax.set_xlabel("Loan term (years)")
    ax.set_ylabel("Approved (%)")
    saved.append(_save(fig, "06_term_vs_approval.png"))

    corr = tmp.corr()["approved"].drop("approved").sort_values(key=abs, ascending=False)
    return {
        "figures": saved,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "top_correlations_with_approval": {k: round(float(v), 3) for k, v in corr.head(5).items()},
        "describe": df[config.NUMERIC_FEATURES].describe().round(1).to_dict(),
    }