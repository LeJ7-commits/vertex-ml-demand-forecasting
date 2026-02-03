# Engineering decisions

## Goal
Build a low-cost, portfolio-grade demand forecasting MLOps platform on Vertex AI.

## Non-goals (for v0)
- No streaming ingestion
- No Feature Store (start with BigQuery feature tables)
- No heavy deep learning

## Key choices
- Model: start with tree-based (fast + cheap)
- Time-based split for forecasting validity
- Conformal prediction intervals as uncertainty layer
- Terraform for reproducible infra
