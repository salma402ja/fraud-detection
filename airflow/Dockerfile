FROM apache/airflow:2.8.1

# Install ML dependencies at build time so they are cached as a Docker layer.
# These packages are needed by fraud_retrain_dag.py at runtime.
RUN pip install --no-cache-dir \
    xgboost \
    scikit-learn \
    scipy \
    pandas \
    joblib
