import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"rows_raw": int(len(df)), "cols_raw": int(df.shape[1])}
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    log["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    if config.ID_COLUMN in df.columns: 
        df = df.drop(columns=[config.ID_COLUMN])
    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log["missing_target_dropped"] = int(df[config.TARGET].isna().sum())
    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)
    asset_cols = [c for c in config.NUMERIC_FEATURES if "assets" in c or "asset_" in c]
    neg = int((df[asset_cols] < 0).sum().sum())
    df[asset_cols] = df[asset_cols].clip(lower=0)
    log["negative_asset_values_clipped"] = neg
    log["rows_clean"] = int(len(df))
    return df, log

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["income_annum"]
    df["total_assets"] = (
        df["residential_assets_value"] + df["commercial_assets_value"]
        + df["luxury_assets_value"] + df["bank_asset_value"]
    )
    df["asset_coverage"] = df["total_assets"] / df["loan_amount"]
    return df

def encode_target(df: pd.DataFrame) -> pd.Series:
    return (df[config.TARGET] == config.POSITIVE_LABEL).astype(int)

def feature_columns() -> tuple[list[str], list[str]]:
    num = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    return num, config.CATEGORICAL_FEATURES

def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num, cat = feature_columns()
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), cat),
    ])

def split(df: pd.DataFrame):
    num, cat = feature_columns()
    X = df[num + cat]
    y = encode_target(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )