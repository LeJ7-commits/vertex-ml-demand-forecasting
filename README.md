# Vertex ML Demand Forecasting (Vertex AI + Conformal Prediction)

## What this repo delivers
- Ingest CSV → BigQuery (raw)
- Feature table in BigQuery (lags/rolling features)
- Training (LightGBM/XGBoost/sklearn) + Conformal prediction intervals
- Orchestration via Vertex AI Pipelines
- Model registration (Vertex Model Registry)
- Deploy to Endpoint + basic online inference client
- Minimal monitoring/runbook

## Repo layout
- infra/terraform: GCS, BigQuery dataset, Artifact Registry, service account
- src/ingest: load raw data into BigQuery
- src/features: feature engineering (BQ SQL + optional Python)
- src/training: trainer + conformal intervals
- pipelines: pipeline definition + compile/run scripts
- docs: architecture + decisions + runbook

## Quickstart (local)
_TODO_

## Quickstart (Vertex)
_TODO_

## Cost controls
- Do not leave endpoints deployed when not demoing.
- Keep dataset small.
- Limit pipeline runs during development.

## Architecture
See `docs/architecture.png`
