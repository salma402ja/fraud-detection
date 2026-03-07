"""
Training script – Fraud Detection model
Generates synthetic data, trains XGBoost, saves model + baseline statistics.
Run: python model/train.py
"""
import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

MODEL_PATH = os.getenv("MODEL_PATH", "model/fraud_model.pkl")
BASELINE_PATH = os.getenv("BASELINE_PATH", "model/baseline_stats.json")
FEATURES = ["amount", "hour", "merchant_category", "distance_from_home", "num_transactions_last_24h"]


def generate_data(n_samples: int = 5000, fraud_rate: float = 0.09, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic transaction data with realistic fraud patterns."""
    np.random.seed(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "amount": np.random.exponential(60, n_legit),
        "hour": np.random.randint(8, 22, n_legit),
        "merchant_category": np.random.randint(0, 5, n_legit),
        "distance_from_home": np.random.exponential(12, n_legit),
        "num_transactions_last_24h": np.random.poisson(3, n_legit),
        "label": 0,
    })

    fraud = pd.DataFrame({
        "amount": np.random.exponential(600, n_fraud),
        "hour": np.random.choice([0, 1, 2, 3, 23], n_fraud),
        "merchant_category": np.random.randint(6, 10, n_fraud),
        "distance_from_home": np.random.exponential(250, n_fraud),
        "num_transactions_last_24h": np.random.poisson(12, n_fraud),
        "label": 1,
    })

    return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


def save_baseline(df: pd.DataFrame, path: str) -> None:
    """Save feature statistics to use as drift detection baseline."""
    stats = {}
    for col in FEATURES:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "p25": float(df[col].quantile(0.25)),
            "p75": float(df[col].quantile(0.75)),
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Baseline statistics saved → {path}")


def train(data_path: str = None, seed: int = 42) -> float:
    # Load or generate data
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded data from {data_path} ({len(df)} rows)")
    else:
        df = generate_data(seed=seed)
        print(f"Generated synthetic data ({len(df)} rows)")

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"Test AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # Save baseline stats for drift detection
    save_baseline(X_train, BASELINE_PATH)

    return auc


if __name__ == "__main__":
    auc = train()
    print(f"\nTraining complete. AUC = {auc:.4f}")
    if auc < 0.85:
        print("WARNING: AUC below threshold (0.85). Check data quality.")
