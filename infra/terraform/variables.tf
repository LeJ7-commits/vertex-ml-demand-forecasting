variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type        = string
  description = "GCP region, e.g. europe-north1"
}

variable "bucket_name" {
  type        = string
  description = "GCS bucket name for Vertex pipeline artifacts"
}

variable "bq_dataset_id" {
  type        = string
  description = "BigQuery dataset id (no dashes), e.g. demand_fcst"
}

variable "ar_repo_id" {
  type        = string
  description = "Artifact Registry repo id, e.g. trainer"
}

variable "labels" {
  type    = map(string)
  default = { env = "dev", app = "vertex-demand-forecasting" }
}
