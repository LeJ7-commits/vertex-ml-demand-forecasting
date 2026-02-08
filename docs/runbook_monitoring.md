# Model Monitoring Runbook

## What is monitored
- Feature drift on:
  - lag_1
  - lag_7
  - roll_mean_7

## Monitoring frequency
- Every 24 hours

## Where to check
Vertex AI Console:
Endpoints → Model Monitoring

## If drift alert occurs
1. Inspect drifted features
2. Compare recent data vs training baseline
3. Retrain pipeline:
   python pipelines/run_pipeline.py ...

## Cost control
- Monitoring job runs once per day
- Disable if not actively using endpoint
