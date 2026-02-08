# Vertex ML Demand Forecasting  
**End-to-end demand forecasting system on Vertex AI with conformal prediction intervals**
Production-style demand forecasting system on Google Cloud Vertex AI with conformal prediction intervals, automated pipelines, deployment, and monitoring.
![CI](https://github.com/LeJ7-commits/vertex-ml-demand-forecasting/actions/workflows/ci.yml/badge.svg)
---

## What this system does

- Ingests raw demand data into BigQuery
- Builds time-series features (lags, rolling statistics)
- Trains gradient boosting models for demand forecasting
- Produces conformal prediction intervals
- Runs as a reproducible Vertex AI pipeline
- Registers models in the Vertex Model Registry
- Deploys models to a managed online endpoint
- Logs predictions for monitoring
- Monitors feature drift to detect data distribution shifts
- Provides a runbook for retraining and operations

---

## Example use case

Retail demand forecasting for:
- Inventory planning
- Stock replenishment
- Promotion impact estimation
- Safety stock decisions using uncertainty intervals

--- 

## Architecture

(TBP)

**Pipeline flow:**
Raw data → BigQuery → Feature table →  
Vertex Training → Conformal calibration →  
Model Registry → Endpoint → Monitoring

--- 

## Tech stack

- Python, scikit-learn, LightGBM
- Google Cloud: BigQuery, Vertex AI, Artifact Registry
- Vertex AI Pipelines (KFP)
- Conformal prediction for uncertainty estimation
- Terraform for infrastructure

---

## Repo layout

- **infra/terraform**  
  GCS, BigQuery dataset, Artifact Registry, service account

- **src/ingest**  
  Load raw data into BigQuery

- **src/features**  
  Feature engineering (SQL + Python)

- **src/training**  
  Model training + conformal prediction intervals

- **pipelines**  
  Vertex pipeline definition and run scripts

- **monitoring**  
  Model monitoring configuration

- **scripts**  
  Operational utilities (baseline export, etc.)

- **docs**  
  Runbooks, and decisions

---

## Quickstart (local training)

```bash
pip install -r requirements.txt
python src/training/train.py
```

## Run on Vertex AI

Set environment variables:

```bash
export PROJECT_ID=your-project
export REGION=europe-north1
export PIPELINE_ROOT=gs://your-bucket/pipeline-root
export GCS_PREFIX=gs://your-bucket/pipeline-artifacts
export IMAGE=your-artifact-registry-image
```

Run the pipeline:

```bash
python pipelines/run_pipeline.py \
  --project_id $PROJECT_ID \
  --region $REGION \
  --pipeline_root $PIPELINE_ROOT \
  --gcs_prefix $GCS_PREFIX \
  --trainer_image $IMAGE \
  --serving_image $IMAGE
```

## Example output

Model metrics from a sample run:

- SMAPE: ~1.9%
- Coverage (90% interval): ~0.80
- Average interval width: ~0.71

This demonstrates calibrated uncertainty using conformal prediction.
  
## Monitoring
- Prediction logging enabled on endpoint
- Baseline dataset exported from training features
- Daily drift checks on key numeric features
- Runbook: `docs/runbook_monitoring.md`

## Cost controls
- Single-replica endpoint
- Manual deployment (no auto-scaling)
- Daily monitoring schedule only
- Small dataset for development
- Delete endpoint when not in use

## Engineering decisions

### Why BigQuery
- Native analytical warehouse
- Handles large retail datasets
- Tight integration with Vertex AI

### Why Vertex AI Pipelines
- Managed orchestration
- Reproducible training runs
- Integrated model registry and endpoints

### Why conformal prediction
- Distribution-free uncertainty estimates
- Stable coverage guarantees
- Practical for real-world forecasting

## Future improvements
- Automated retraining triggers
- Feature store integration
- Batch prediction pipeline
- Alerting via Cloud Monitoring
