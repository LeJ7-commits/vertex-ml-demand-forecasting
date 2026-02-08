from google.cloud import bigquery

PROJECT_ID = "vertex-demand-260203-13078"
DATASET = "demand_fcst"
TABLE = "features_demand_daily"
OUTPUT_URI = "gs:// vertex-demand-260203-13078-artifacts/baseline/features-*.parquet"

client = bigquery.Client(project=PROJECT_ID)

job_config = bigquery.job.ExtractJobConfig(
    destination_format="PARQUET"
)

extract_job = client.extract_table(
    f"{PROJECT_ID}.{DATASET}.{TABLE}",
    OUTPUT_URI,
    job_config=job_config,
)

extract_job.result()
print("Baseline export complete.")
