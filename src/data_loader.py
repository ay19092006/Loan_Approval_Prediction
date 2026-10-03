import numpy as np
import pandas as pd
from src import config

def generate_synthetic(n: int = 4269, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dependents = rng.integers(0, 6, n)
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.5, 0.5])
    self_emp = rng.choice(["Yes", "No"], n, p=[0.5, 0.5])
    income = rng.integers(2_00_000, 99_00_000, n)
    loan_amount = (income * rng.uniform(0.5, 4.5, n)).astype(int)
    loan_amount = np.clip(loan_amount, 3_00_000, 39_50_00_000)
    loan_term = rng.integers(2, 21, n)
    cibil = rng.integers(300, 900, n)
    res_assets = rng.integers(-1_00_000, 29_10_00_000, n)
    com_assets = rng.integers(0, 19_40_00_000, n)
    lux_assets = rng.integers(3_00_000, 39_20_00_000, n)
    bank_assets = rng.integers(0, 14_70_00_000, n)
    total_assets = res_assets + com_assets + lux_assets + bank_assets
    score = (
        (cibil - 600) / 100.0 * 2.6
        - (loan_amount / income) * 0.35
        + np.clip(total_assets / loan_amount, 0, 5) * 0.10
        - (loan_term - 10) * 0.03
        - dependents * 0.05
        + (education == "Graduate") * 0.15
        + rng.normal(0, 0.9, n)
        + 2.0
    )
    approved = score > 0
    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        " no_of_dependents": dependents,
        " education": " " + pd.Series(education),
        " self_employed": " " + pd.Series(self_emp),
        " income_annum": income,
        " loan_amount": loan_amount,
        " loan_term": loan_term,
        " cibil_score": cibil,
        " residential_assets_value": res_assets,
        " commercial_assets_value": com_assets,
        " luxury_assets_value": lux_assets,
        " bank_asset_value": bank_assets,
        " loan_status": np.where(approved, " Approved", " Rejected"),
    })
    for col in [" cibil_score", " income_annum", " education"]:
        idx = rng.choice(n, size=int(0.01 * n), replace=False)
        df.loc[idx, col] = np.nan
    dupes = df.sample(25, random_state=seed)
    return pd.concat([df, dupes], ignore_index=True)

def load_raw() -> pd.DataFrame:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if not config.RAW_FILE.exists():
        print(f"[data] {config.RAW_FILE.name} not found -> generating SYNTHETIC stand-in.")
        generate_synthetic().to_csv(config.RAW_FILE, index=False)
        (config.DATA_RAW / ".synthetic").touch()
    else:
        print(f"[data] Loading {config.RAW_FILE}")
    return pd.read_csv(config.RAW_FILE)

def is_synthetic() -> bool:
    marker = config.DATA_RAW / ".synthetic"
    return marker.exists()