import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, Request

app = FastAPI()

AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/model")

def _load_model():
    # support common layouts:
    candidates = [
        os.path.join(AIP_MODEL_DIR, "model", "model.joblib"),
        os.path.join(AIP_MODEL_DIR, "model.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p), p
    raise FileNotFoundError(f"model.joblib not found in AIP_MODEL_DIR={AIP_MODEL_DIR}. Tried: {candidates}")

MODEL, MODEL_PATH = _load_model()

@app.get(AIP_HEALTH_ROUTE)
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    payload = await request.json()
    instances = payload.get("instances", payload)

    # instances can be list[dict] or dict
    if isinstance(instances, dict):
        df = pd.DataFrame([instances])
    else:
        df = pd.DataFrame(instances)

    preds = MODEL.predict(df)
    return {"predictions": preds.tolist()}
