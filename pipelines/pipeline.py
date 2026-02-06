from kfp import dsl
from kfp.dsl import Artifact, Output, Input, component
from typing import Optional

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "google-cloud-storage>=2.16.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "db-dtypes>=1.0.0",
        "pyarrow>=12.0.0",
    ],
)
def extract_features_to_gcs(
    project_id: str,
    dataset: str,
    table: str,
    gcs_uri_prefix: str,
    out_uri: Output[Artifact],
):
    """
    Exports BigQuery table to GCS as Parquet shards using BigQuery Extract Job.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    src_table = f"{project_id}.{dataset}.{table}"
    dest_uri = f"{gcs_uri_prefix}/bq_export/features-*.parquet"

    job_config = bigquery.job.ExtractJobConfig(destination_format="PARQUET")
    job = client.extract_table(src_table, dest_uri, job_config=job_config)
    job.result()

    with open(out_uri.path, "w") as f:
        f.write(dest_uri)



@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "db-dtypes>=1.0.0",
        "pyarrow>=12.0.0",
    ],
)
def train_local_component(
    project_id: str,
    dataset: str,
    table: str,
    gcs_artifacts_prefix: str,
    alpha: float,
    limit: int,
    model_dir: Output[Artifact],
    metrics: Output[Artifact],
):
    """
    Runs a lightweight training job inside a pipeline component container.
    Reads from BigQuery directly (cheaper/cleaner than staging right now).
    Writes artifacts to GCS + returns artifact URIs.
    """
    import json
    import os
    import numpy as np
    import pandas as pd
    from google.cloud import bigquery
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def smape(y_true, y_pred, eps=1e-8):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
        return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

    def conformal_quantile(y_true, y_pred, alpha=0.1):
        res = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
        return float(np.quantile(res, 1.0 - alpha))

    def interval_coverage(y_true, lo, hi):
        y_true = np.asarray(y_true, float)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))

    client = bigquery.Client(project=project_id)
    sql = f"""
    SELECT
      date, store_id, item_id,
      y,
      dow, week, month, quarter, year,
      sin_doy, cos_doy,
      lag_1, lag_7, lag_14, lag_28,
      roll_mean_7, roll_mean_14, roll_mean_28,
      roll_std_28
    FROM `{project_id}.{dataset}.{table}`
    """
    if limit and limit > 0:
        sql += f"\nLIMIT {int(limit)}"

    df = client.query(sql).to_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # time split by unique dates
    unique_dates = sorted(df["date"].dropna().unique().tolist())
    n = len(unique_dates)
    train_end = unique_dates[int(n * 0.70) - 1]
    cal_end = unique_dates[int(n * 0.85) - 1]

    train_df = df[df["date"] <= train_end]
    cal_df = df[(df["date"] > train_end) & (df["date"] <= cal_end)]
    test_df = df[df["date"] > cal_end]

    # encode categories cheaply
    for part in (train_df, cal_df, test_df):
        part["store_id_enc"], _ = pd.factorize(part["store_id"])
        part["item_id_enc"], _ = pd.factorize(part["item_id"])

    feature_cols = [
        "store_id_enc", "item_id_enc",
        "dow", "week", "month", "quarter", "year",
        "sin_doy", "cos_doy",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
        "roll_std_28",
    ]

    X_train = train_df[feature_cols].astype(float).values
    y_train = train_df["y"].astype(float).values
    X_cal = cal_df[feature_cols].astype(float).values
    y_cal = cal_df["y"].astype(float).values
    X_test = test_df[feature_cols].astype(float).values
    y_test = test_df["y"].astype(float).values

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)

    cal_pred = model.predict(X_cal)
    q = conformal_quantile(y_cal, cal_pred, alpha=alpha)

    test_pred = model.predict(X_test)
    lo = test_pred - q
    hi = test_pred + q

    mae = float(mean_absolute_error(y_test, test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    s = smape(y_test, test_pred)
    cov = interval_coverage(y_test, lo, hi)
    avg_width = float(np.mean(hi - lo))

    out = {
        "split": {
            "train_end_date": str(pd.to_datetime(train_end).date()),
            "cal_end_date": str(pd.to_datetime(cal_end).date()),
            "n_train": int(len(train_df)),
            "n_cal": int(len(cal_df)),
            "n_test": int(len(test_df)),
        },
        "conformal": {"alpha": float(alpha), "q_abs_resid": float(q)},
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "smape": s,
            "coverage": cov,
            "avg_interval_width": avg_width,
        },
        "features": feature_cols,
        "model": "HistGradientBoostingRegressor",
    }

    import os
    os.makedirs(model_dir.path, exist_ok=True)
    dump(model, os.path.join(model_dir.path, "model.joblib"))
    with open(metrics.path, "w") as f:
        json.dump(out, f, indent=2)

    # Also print metrics for logs
    print(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))



@dsl.pipeline(name="demand-forecasting-vtx-pipeline")
def pipeline(
    project_id: str,
    dataset: str = "demand_fcst",
    features_table: str = "features_demand_daily",
    gcs_prefix: str = "",
    alpha: float = 0.1,
    limit: int = 0,
):
    extract_features_to_gcs(
        project_id=project_id,
        dataset=dataset,
        table=features_table,
        gcs_uri_prefix=gcs_prefix,
    )

    train_local_component(
        project_id=project_id,
        dataset=dataset,
        table=features_table,
        gcs_artifacts_prefix=gcs_prefix,
        alpha=alpha,
        limit=limit,
    )
