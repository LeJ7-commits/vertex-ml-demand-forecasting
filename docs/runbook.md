# Runbook

## Safe defaults
- Keep endpoints undeployed when not testing
- Use small dev datasets
- Prefer batch prediction until final demo

## Troubleshooting checklist
- Auth: `gcloud auth application-default login`
- Project: `gcloud config set project ...`
- Vertex API enabled
- BigQuery dataset exists
- GCS bucket exists
