output "BUCKET_NAME" {
  value = google_storage_bucket.pipeline_artifacts.name
}

output "BQ_DATASET" {
  value = google_bigquery_dataset.demand.dataset_id
}

output "AR_REPO" {
  value = google_artifact_registry_repository.trainer_repo.repository_id
}

output "SERVICE_ACCOUNT_EMAIL" {
  value = google_service_account.pipeline_runner.email
}
