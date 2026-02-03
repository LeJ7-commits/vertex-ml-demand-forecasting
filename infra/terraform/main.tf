provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Enable required APIs (prevents "API not enabled" surprises)
# Terraform samples often assume this is enabled; we enforce it.  :contentReference[oaicite:2]{index=2}
resource "google_project_service" "aiplatform" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "bigquery" {
  service            = "bigquery.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# --- Service account for Vertex AI Pipelines runs
resource "google_service_account" "pipeline_runner" {
  account_id   = "vertex-pipeline-runner"
  display_name = "Vertex AI pipeline runner (dev)"
}

# --- GCS bucket for pipeline artifacts (Vertex AI Pipelines artifact store) :contentReference[oaicite:3]{index=3}
resource "google_storage_bucket" "pipeline_artifacts" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  versioning {
    enabled = true
  }

  labels = var.labels

  depends_on = [google_project_service.storage]
}

# Bucket IAM: allow the pipeline SA to read/write artifacts
resource "google_storage_bucket_iam_member" "pipeline_bucket_object_admin" {
  bucket = google_storage_bucket.pipeline_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# --- BigQuery dataset for raw + features
resource "google_bigquery_dataset" "demand" {
  dataset_id = var.bq_dataset_id
  location   = var.region
  labels     = var.labels

  depends_on = [google_project_service.bigquery]
}

# Project-level IAM for BigQuery usage (query + write tables)
resource "google_project_iam_member" "pipeline_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_project_iam_member" "pipeline_bq_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# --- Artifact Registry repository for training container images
# Terraform resource is standard for AR repos. :contentReference[oaicite:4]{index=4}
resource "google_artifact_registry_repository" "trainer_repo" {
  location      = var.region
  repository_id = var.ar_repo_id
  description   = "Training images for vertex demand forecasting"
  format        = "DOCKER"
  labels        = var.labels

  depends_on = [google_project_service.artifactregistry]
}

# Allow SA to pull/push images
resource "google_project_iam_member" "pipeline_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# --- Vertex AI permissions for running pipelines and managing model resources
resource "google_project_iam_member" "pipeline_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"

  depends_on = [google_project_service.aiplatform]
}
