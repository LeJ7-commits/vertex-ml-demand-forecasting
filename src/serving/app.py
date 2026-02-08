import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

MODEL = None
MODEL_PATH = None


def _download_from_gcs(gcs_uri: str, local_path: str) -> str:
    # gcs_uri like gs://bucket/path/to/model.joblib
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// uri: {gcs_uri}")

    _, rest = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = rest.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def _resolve_model_path() -> str:
    # 1) If Vertex mounts a local model directory
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    if aip_model_dir and os.path.isdir(aip_model_dir):
        candidates = [
            os.path.join(aip_model_dir, "model.joblib"),
            os.path.join(aip_model_dir, "model", "model.joblib"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    # 2) Otherwise use AIP_STORAGE_URI (GCS path to artifact-uri)
    # If artifact-uri is a folder, model.joblib should be under it (or under /model/)
    aip_storage_uri = os.environ.get("AIP_STORAGE_URI")
    if aip_storage_uri:
        if aip_storage_uri.endswith("/"):
            base = aip_storage_uri[:-1]
        else:
            base = aip_storage_uri

        # Try both common layouts
        gcs_candida_
