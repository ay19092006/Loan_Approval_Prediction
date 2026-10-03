import json
import joblib
from src import config
from src.data_loader import load_raw
from src.eda import run_eda
from src.evaluate import evaluate_models, feature_importance
from src.preprocessing import add_features, clean, split
from src.train import cross_validate, fit_all, get_models

def main() -> None:
    for d in (config.DATA_PROCESSED, config.MODELS_DIR, config.METRICS_DIR, config.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df, clean_log = clean(raw)
    df.to_csv(config.CLEAN_FILE, index=False)
    print(f"[clean] {clean_log}")
    eda_info = run_eda(df)
    df_fe = add_features(df)
    X_train, X_test, y_train, y_test = split(df_fe)
    print(f"[split] train={len(X_train)} test={len(X_test)} "
          f"(approved share train={y_train.mean():.3f}, test={y_test.mean():.3f})")
    models = get_models()
    cv = cross_validate(models, X_train, y_train)
    models = fit_all(models, X_train, y_train)
    ev = evaluate_models(models, X_test, y_test)
    imp = feature_importance(models["Random Forest"])
    candidates = {k: v for k, v in ev["metrics"].items() if not k.startswith("Baseline")}
    best = max(candidates, key=lambda k: candidates[k]["f1"])
    joblib.dump(models[best], config.MODEL_FILE)
    print(f"[save] best model: {best} -> {config.MODEL_FILE}")
    results = {
        "dataset_source": config.DATASET_SOURCE,
        "cleaning": clean_log,
        "eda": eda_info,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)),
                  "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE},
        "cross_validation": cv,
        "evaluation": ev,
        "feature_importance": imp,
        "best_model": best,
    }
    config.METRICS_FILE.write_text(json.dumps(results, indent=2))
    print(f"[done] metrics -> {config.METRICS_FILE}")

if __name__ == "__main__":
    main()