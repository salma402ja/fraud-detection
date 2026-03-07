"""
Airflow DAG – Fraud Detection Model Nightly Retraining
=======================================================
Scenario: the fraud detection model is retrained every night.
Before retraining, the pipeline checks for data drift by comparing
the distribution of recent transactions to the training baseline.

DAG flow:
  extract_transactions
       ↓
  validate_data          ← quality gate (stops if data is bad)
       ↓
  check_data_drift       ← drift gate (branches: retrain vs skip)
       ↓                          ↓
  preprocess_features       end_no_drift
       ↓
  train_model
       ↓
  evaluate_auc           ← quality gate (stops if AUC < 0.85)
       ↓
  save_to_registry

Key properties:
  - No external observability tool required (scipy only)
  - Quality gates stop the pipeline on bad data or a degraded model
  - Full lineage: data → model → registry
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ── Paths & thresholds ─────────────────────────────────────────────────────
DATA_PATH       = os.getenv("DATA_PATH",       "/opt/airflow/data/transactions.csv")
PROCESSED_PATH  = os.getenv("PROCESSED_PATH",  "/opt/airflow/data/transactions_processed.csv")
MODEL_PATH      = os.getenv("MODEL_PATH",       "/opt/airflow/model/fraud_model.pkl")
CANDIDATE_PATH  = os.getenv("CANDIDATE_PATH",   "/opt/airflow/model/fraud_model_candidate.pkl")
BASELINE_PATH   = os.getenv("BASELINE_PATH",    "/opt/airflow/model/baseline_stats.json")

FEATURES        = ["amount", "hour", "merchant_category", "distance_from_home", "num_transactions_last_24h"]
MIN_ROWS        = 500       # minimum acceptable dataset size
MAX_NULL_PCT    = 0.05      # max fraction of nulls allowed
AUC_THRESHOLD   = 0.85      # minimum AUC to promote the model
DRIFT_P_VALUE   = 0.05      # KS-test p-value threshold


# ══════════════════════════════════════════════════════════════════════════
# Task functions
# ══════════════════════════════════════════════════════════════════════════

def extract_transactions(**ctx):
    """
    Simulate loading the last 30 days of transactions from a database.
    In production: replace with a SQL query or S3 download.
    """
    log = logging.getLogger(__name__)
    np.random.seed(int(datetime.utcnow().timestamp()) % 10_000)

    n = np.random.randint(700, 1200)
    n_fraud = int(n * np.random.uniform(0.07, 0.12))
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "amount":                    np.random.exponential(60, n_legit),
        "hour":                      np.random.randint(8, 22, n_legit),
        "merchant_category":         np.random.randint(0, 5, n_legit),
        "distance_from_home":        np.random.exponential(12, n_legit),
        "num_transactions_last_24h": np.random.poisson(3, n_legit),
        "label": 0,
    })
    fraud = pd.DataFrame({
        "amount":                    np.random.exponential(600, n_fraud),
        "hour":                      np.random.choice([0, 1, 2, 3, 23], n_fraud),
        "merchant_category":         np.random.randint(6, 10, n_fraud),
        "distance_from_home":        np.random.exponential(250, n_fraud),
        "num_transactions_last_24h": np.random.poisson(12, n_fraud),
        "label": 1,
    })

    df = pd.concat([legit, fraud]).sample(frac=1).reset_index(drop=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    log.info(f"Extracted {len(df)} transactions ({n_fraud} fraud, {n_legit} legit) → {DATA_PATH}")
    ctx["ti"].xcom_push(key="row_count", value=len(df))


def validate_data(**ctx):
    """
    Quality gate #1: reject the dataset if it is too small or too dirty.
    Raises ValueError to fail the task and stop the pipeline.
    """
    log = logging.getLogger(__name__)
    df = pd.read_csv(DATA_PATH)

    row_count = len(df)
    null_pct  = df[FEATURES].isnull().mean().max()

    log.info(f"Rows: {row_count}  |  Max null ratio: {null_pct:.2%}")

    if row_count < MIN_ROWS:
        raise ValueError(f"Not enough data: {row_count} rows (minimum {MIN_ROWS})")

    if null_pct > MAX_NULL_PCT:
        raise ValueError(f"Too many nulls: {null_pct:.2%} (maximum {MAX_NULL_PCT:.0%})")

    log.info("Data validation passed ✓")


def check_data_drift(**ctx):
    """
    Branch task: detect whether the new data has drifted from training baseline.

    Method: Kolmogorov-Smirnov two-sample test on each numeric feature.
            No external observability tool – only scipy.stats.

    Returns:
        'preprocess_features' → drift detected, proceed with retraining
        'end_no_drift'        → no drift, skip retraining
    """
    log = logging.getLogger(__name__)

    # If no baseline exists (first run), always retrain
    if not os.path.exists(BASELINE_PATH):
        log.info("No baseline found (first run) → retraining")
        return "preprocess_features"

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    df_new = pd.read_csv(DATA_PATH)

    drift_detected = False
    drift_features = []

    for feature in ["amount", "distance_from_home", "num_transactions_last_24h"]:
        # Reconstruct a reference sample from baseline statistics
        b = baseline[feature]
        ref_sample = np.random.normal(b["mean"], b["std"], size=len(df_new))
        ref_sample = np.clip(ref_sample, 0, None)

        ks_stat, p_value = stats.ks_2samp(ref_sample, df_new[feature].dropna())
        log.info(f"  KS [{feature}]: stat={ks_stat:.4f}  p={p_value:.4f}")

        if p_value < DRIFT_P_VALUE:
            log.warning(f"  Drift detected on '{feature}' (p={p_value:.4f} < {DRIFT_P_VALUE})")
            drift_detected = True
            drift_features.append(feature)

    if drift_detected:
        log.info(f"Drift detected on: {drift_features} → retraining")
        return "preprocess_features"
    else:
        log.info("No significant drift detected → skipping retraining")
        return "end_no_drift"


def preprocess_features(**ctx):
    """Clip outliers and write processed dataset."""
    log = logging.getLogger(__name__)
    df = pd.read_csv(DATA_PATH)

    df["amount"]                    = df["amount"].clip(0, 10_000)
    df["distance_from_home"]        = df["distance_from_home"].clip(0, 2_000)
    df["num_transactions_last_24h"] = df["num_transactions_last_24h"].clip(0, 50)
    df["hour"]                      = df["hour"].clip(0, 23)
    df["merchant_category"]         = df["merchant_category"].clip(0, 9)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    log.info(f"Preprocessed data saved → {PROCESSED_PATH}")


def train_model(**ctx):
    """Train XGBoost classifier on the last 30 days of transactions."""
    log = logging.getLogger(__name__)
    df = pd.read_csv(PROCESSED_PATH)

    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    log.info(f"Training complete. Test AUC = {auc:.4f}")

    # Save candidate (not yet production)
    os.makedirs(os.path.dirname(CANDIDATE_PATH), exist_ok=True)
    joblib.dump(clf, CANDIDATE_PATH)

    ctx["ti"].xcom_push(key="auc", value=auc)
    log.info(f"Candidate model saved → {CANDIDATE_PATH}")


def evaluate_auc(**ctx):
    """
    Quality gate #2: promote the model only if AUC >= threshold.
    Raises ValueError to fail the task and block promotion.
    """
    log = logging.getLogger(__name__)
    auc = ctx["ti"].xcom_pull(key="auc", task_ids="train_model")

    log.info(f"Evaluating model: AUC={auc:.4f}  threshold={AUC_THRESHOLD}")

    if auc < AUC_THRESHOLD:
        raise ValueError(
            f"Model AUC too low: {auc:.4f} < {AUC_THRESHOLD}. "
            "Candidate model rejected – current production model kept."
        )

    log.info(f"AUC validation passed ✓  ({auc:.4f} ≥ {AUC_THRESHOLD})")


def save_to_registry(**ctx):
    """
    Promote the candidate model to production.
    In production: upload to S3 / MLflow Model Registry / artifact store.
    """
    log = logging.getLogger(__name__)
    auc = ctx["ti"].xcom_pull(key="auc", task_ids="train_model")

    shutil.copy(CANDIDATE_PATH, MODEL_PATH)
    log.info(f"Candidate promoted to production: {MODEL_PATH}  (AUC={auc:.4f})")

    # Write metadata file (acts as a lightweight model registry)
    meta_path = MODEL_PATH.replace(".pkl", "_meta.json")
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "auc": round(auc, 4),
        "features": FEATURES,
        "auc_threshold": AUC_THRESHOLD,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Metadata saved → {meta_path}")

    # --- Extend here in production ---
    # import mlflow
    # mlflow.log_metric("auc", auc)
    # mlflow.log_artifact(MODEL_PATH)
    #
    # import boto3
    # boto3.client("s3").upload_file(MODEL_PATH, "my-bucket", "models/fraud_model.pkl")


# ══════════════════════════════════════════════════════════════════════════
# DAG definition
# ══════════════════════════════════════════════════════════════════════════

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="fraud_model_nightly_retrain",
    description="Nightly retraining of the fraud detection model with drift detection",
    schedule_interval=None,   # TODO (Part 5.4): replace None with the correct cron expression for 19:00 local time
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["fraud", "ml", "production", "mlops"],
) as dag:

    t_extract    = PythonOperator(task_id="extract_transactions", python_callable=extract_transactions)
    t_validate   = PythonOperator(task_id="validate_data",        python_callable=validate_data)

    t_drift      = BranchPythonOperator(task_id="check_data_drift", python_callable=check_data_drift)

    t_preprocess = PythonOperator(task_id="preprocess_features",  python_callable=preprocess_features)
    t_train      = PythonOperator(task_id="train_model",          python_callable=train_model)
    t_evaluate   = PythonOperator(task_id="evaluate_auc",         python_callable=evaluate_auc)
    t_save       = PythonOperator(task_id="save_to_registry",     python_callable=save_to_registry)

    t_no_drift   = EmptyOperator(task_id="end_no_drift")   # branch: no retrain needed

    # ── Pipeline graph ─────────────────────────────────────────────────────
    t_extract >> t_validate >> t_drift
    t_drift >> t_preprocess >> t_train >> t_evaluate >> t_save
    t_drift >> t_no_drift
