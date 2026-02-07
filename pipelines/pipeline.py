from kfp import dsl
from kfp.dsl import Artifact, Output, component
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "google-cloud-storage>=2.16.0",
    ],
)
def extract_features_to_gcs(
    project_id: str,
    dataset: str,
    table: str,
    gcs_uri_prefix: str,
    out_uri: Output[Artifact],
):
    """
    Exports BigQuery table to GCS as Parquet shards.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    src_table = f"{project_id}.{dataset}.{table}"
    dest_uri = f"{gcs_uri_prefix}/bq_export/features-*.parquet"

    job_config = bigquery.job.ExtractJobConfig(destination_format="PARQUET")
    job = client.extract_table(src_table, dest_uri, job_config=job_config)
    job.result()

    with open(out_uri.path, "w") as f:
        f.write(dest_uri)


@dsl.pipeline(name="demand-forecasting-vtx-pipeline")
def pipeline(
    project_id: str,
    region: str,
    dataset: str = "demand_fcst",
    features_table: str = "features_demand_daily",
    gcs_prefix: str = "",
    trainer_image: str = "",
    serving_image: str = "",                 # NEW
    model_display_name: str = "demand-fcst", # NEW
    parent_model: str = "",                  # NEW
    alpha: float = 0.1,
    limit: int = 0,
):
    # Step 1: extract features
    extract = extract_features_to_gcs(
        project_id=project_id,
        dataset=dataset,
        table=features_table,
        gcs_uri_prefix=gcs_prefix,
    )

    # Step 2: run custom container training job
    train = CustomTrainingJobOp(
        display_name="train-demand-model",
        project=project_id,
        location=region,
        base_output_directory=f"{gcs_prefix}/runs/{dsl.PIPELINE_JOB_NAME_PLACEHOLDER}/train",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": trainer_image,
                "args": [
                    "--project_id", project_id,
                    "--dataset", dataset,
                    "--table", features_table,
                    "--alpha", str(alpha),
                    "--limit", str(limit),
                    "--features_uri", extract.outputs["out_uri"],
                ],
            },
        }],
    )

    train.after(extract)
