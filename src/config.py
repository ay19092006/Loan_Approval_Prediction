from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"
RAW_FILE = DATA_RAW / "loan_approval_dataset.csv"
CLEAN_FILE = DATA_PROCESSED / "loan_clean.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = METRICS_DIR / "results.json"
TARGET = "loan_status"
ID_COLUMN = "loan_id"
POSITIVE_LABEL = "Approved"
NEGATIVE_LABEL = "Rejected"
NUMERIC_FEATURES = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
ENGINEERED_FEATURES = ["loan_to_income", "total_assets", "asset_coverage"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DATASET_SOURCE = (
    "Loan Approval Prediction Dataset (Kaggle, by Archit Sharma) - "
    "https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset"
)
