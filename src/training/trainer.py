import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from google.cloud import bigquery
from google.cloud import storage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.training.conformal import conformal_quantile, conformal_interval, interval_coverage


# -----------------------------
# Config / helpers
# -----------------------------
@dataclass
class SplitConfig:
    train_frac: float = 0.70
    cal_frac: float = 0.15
    test_frac: float = 0.15


def is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def upload_file_to_gcs(local_path: str, gcs_dir: str) -> str:
    """
    Uploads local_path to gcs_dir (a gs://bucket/prefix directory).
    Returns the full gs:// URI of the uploaded object.
    """
    u = urlparse(gcs_dir)
    bucket_name = u.netloc
    prefix = u.path.lstrip("/").rstrip("/")

    filename = os.path.basename(local_path)
    blob_name = f"{prefix}/{filename}" if prefix else filename

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    return f"gs://{bucket_name}/{blob_name}"


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def date_splits(unique_dates: List[pd.Timestamp], cfg: SplitConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    n = len(unique_dates)
    if n < 3:
        raise ValueError(f"Not enough unique dates for splitting: {n} (need >= 3)")

    # Small-data fallback: ensure non-empty cal/test if possible
    if n < 10:
        train_end = unique_dates[n - 3]
        cal_end = unique_dates[n - 2]
        return train_end, cal_end

    train_end_idx = int(np.floor(n * cfg.train_frac)) - 1
    cal_end_idx = int(np.floor(n * (cfg.train_frac + cfg.cal_frac))) - 1

    train_end_idx = int(np.clip(train_end_idx, 0, n - 3))
    cal_end_idx = int(np.clip(cal_end_idx, train_end_idx + 1, n - 2))

    return unique_dates[train_end_idx], unique_dates[cal_end_idx]


def load_features_from_bq(project_id: str, dataset: str, table: str, limit: int | None = None) -> pd.DataFrame:
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
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    df = client.query(sql).to_dataframe(create_bqstorage_client=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_features_from_parquet(features_uri: str) -> pd.DataFrame:
    # features_uri like gs://bucket/path/bq_export/
    dataset = ds.dataset(features_uri, format="parquet")  # directory supported
    table = dataset.to_table()
    df = table.to_pandas()
    return df


def prep_xy(df: pd.DataFrame):
    id_cols = ["date", "store_id", "item_id"]
    y = df["y"].astype(float).values

    feature_cols = [
        "store_id_enc",
        "item_id_enc",
        "dow",
        "week",
        "month",
        "quarter",
        "year",
        "sin_doy",
        "cos_doy",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "roll_mean_7",
        "roll_mean_14",
        "roll_mean_28",
        "roll_std_28",
    ]
    X = df[feature_cols].astype(float).values
    meta = df[id_cols].copy()
    return X, y, meta, feature_cols


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--dataset", default="demand_fcst")
    ap.add_argument("--table", default="features_demand_daily")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--features_uri", default="", help="Optional: gs://.../bq_export/ directory")
    args = ap.parse_args()

    # Vertex injects AIP_MODEL_DIR. It may be a gs:// URI.
    vertex_model_dir = os.environ.get("AIP_MODEL_DIR", "")
    if vertex_model_dir:
        args.artifacts_dir = vertex_model_dir

    # Only create local directories; NEVER os.makedirs(gs://...)
    if not is_gcs(args.artifacts_dir):
        os.makedirs(args.artifacts_dir, exist_ok=True)

    print("AIP_MODEL_DIR=" + vertex_model_dir)
    print("ARTIFACTS_DIR=" + args.artifacts_dir)

    # -----------------------------
    # Load features
    # -----------------------------
    if args.features_uri and args.features_uri.startswith("gs://"):
        df = load_features_from_parquet(args.features_uri)
        if args.limit and args.limit > 0 and len(df) > args.limit:
            df = df.sample(n=int(args.limit), random_state=42)
    else:
        df = load_features_from_bq(
            project_id=args.project_id,
            dataset=args.dataset,
            table=args.table,
            limit=(args.limit if args.limit > 0 else None),
        )

    if df.empty:
        raise ValueError("Loaded features dataframe is empty. Check your source table / parquet path.")

    df = df.sort_values("date").reset_index(drop=True)

    # Factorize ONCE to avoid encoding drift across splits
    df["store_id_enc"], _ = pd.factorize(df["store_id"])
    df["item_id_enc"], _ = pd.factorize(df["item_id"])

    unique_dates = sorted(df["date"].dropna().unique().tolist())
    train_end, cal_end = date_splits(unique_dates, SplitConfig())

    train_df = df[df["date"] <= train_end]
    cal_df = df[(df["date"] > train_end) & (df["date"] <= cal_end)]
    test_df = df[df["date"] > cal_end]

    if len(cal_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty calibration or test split. n_train={len(train_df)} n_cal={len(cal_df)} n_test={len(test_df)}. "
            "Increase date span / reduce limit / adjust splitting."
        )

    X_train, y_train, _, feature_cols = prep_xy(train_df)
    X_cal, y_cal, _, _ = prep_xy(cal_df)
    X_test, y_test, meta_test, _ = prep_xy(test_df)

    # -----------------------------
    # Train + conformal
    # -----------------------------
    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_depth=8,
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)

    cal_pred = model.predict(X_cal)
    q = conformal_quantile(y_cal, cal_pred, alpha=args.alpha)

    test_pred = model.predict(X_test)
    lo, hi = conformal_interval(test_pred, q=q)

    mae = float(mean_absolute_error(y_test, test_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    s = smape(y_test, test_pred)
    cov = interval_coverage(y_test, lo, hi)
    avg_width = float(np.mean(hi - lo))

    metrics = {
        "split": {
            "train_end_date": str(pd.to_datetime(train_end).date()),
            "cal_end_date": str(pd.to_datetime(cal_end).date()),
            "n_train": int(len(train_df)),
            "n_cal": int(len(cal_df)),
            "n_test": int(len(test_df)),
        },
        "conformal": {"alpha": float(args.alpha), "q_abs_resid": float(q)},
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "smape": s,
            "coverage": cov,
            "avg_interval_width": avg_width,
        },
        "features": feature_cols,
        "model": "HistGradientBoostingRegressor",
        "data_source": ("parquet" if (args.features_uri and args.features_uri.startswith("gs://")) else "bigquery"),
        "features_uri": args.features_uri,
    }

    # Predictions dataframe (THIS WAS MISSING IN YOUR BROKEN VERSION)
    pred_out = meta_test.copy()
    pred_out["y_true"] = y_test
    pred_out["y_pred"] = test_pred
    pred_out["y_lo"] = lo
    pred_out["y_hi"] = hi

    # -----------------------------
    # Save artifacts (robust for Vertex)
    # -----------------------------
    out_dir = args.artifacts_dir  # local path OR gs://
    local_dir = "/tmp/artifacts"
    os.makedirs(local_dir, exist_ok=True)

    local_model = os.path.join(local_dir, "model.joblib")
    local_metrics = os.path.join(local_dir, "metrics.json")
    local_preds = os.path.join(local_dir, "predictions.csv")

    joblib.dump(model, local_model)
    with open(local_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    pred_out.to_csv(local_preds, index=False)

    if is_gcs(out_dir):
        model_path = upload_file_to_gcs(local_model, out_dir)
        metrics_path = upload_file_to_gcs(local_metrics, out_dir)
        pred_path = upload_file_to_gcs(local_preds, out_dir)
    else:
        model_path = os.path.join(out_dir, "model.joblib")
        metrics_path = os.path.join(out_dir, "metrics.json")
        pred_path = os.path.join(out_dir, "predictions.csv")
        joblib.dump(model, model_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        pred_out.to_csv(pred_path, index=False)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {pred_path}")

    # Useful for log scraping
    print("METRICS_JSON=" + json.dumps(metrics))


if __name__ == "__main__":
    main()
