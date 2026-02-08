import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

MODEL = None
MODEL_PATH = None


def _download_from_gcs(gcs_uri: str, local_path: str) -> str:
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
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    if aip_model_dir and os.path.isdir(aip_model_dir):
        candidates = [
            os.path.join(aip_model_dir, "model.joblib"),
            os.path.join(aip_model_dir, "model", "model.joblib"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    aip_storage_uri = os.environ.get("AIP_STORAGE_URI")
    if aip_storage_uri:
        base = aip_storage_uri[:-1] if aip_storage_uri.endswith("/") else aip_storage_uri
        gcs_candidates = [
            base + "/model.joblib",
            base + "/model/model.joblib",
        ]
        for gcs_path in gcs_candidates:
            try:
                return _download_from_gcs(gcs_path, "/tmp/model/model.joblib")
            except Exception:
                pass
        raise FileNotFoundError(
            f"Could not find model.joblib in AIP_STORAGE_URI={aip_storage_uri}. Tried: {gcs_candidates}"
        )

    fallback = os.environ.get("MODEL_PATH")
    if fallback and os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Could not locate model.joblib. "
        f"AIP_MODEL_DIR={aip_model_dir} (exists={bool(aip_model_dir and os.path.isdir(aip_model_dir))}), "
        f"AIP_STORAGE_URI={os.environ.get('AIP_STORAGE_URI')}"
    )


def _load_model_once():
    global MODEL, MODEL_PATH
    if MODEL is None:
        MODEL_PATH = _resolve_model_path()
        MODEL = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup():
    _load_model_once()


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    _load_model_once()
    import pandas as pd
    X = pd.DataFrame(req.instances)
    preds = MODEL.predict(X)
    return {"predictions": preds.tolist(), "model_path": MODEL_PATH}
