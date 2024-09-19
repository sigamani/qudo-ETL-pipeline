variable "project_id" {
  description = "The project ID to deploy resources into"
  default     = "clustered-cream"
}

variable "container_image" {
  description = "The container image"
  default     = "europe-west2-docker.pkg.dev/clustered-cream/ak-dagster/poc:latest"
}

variable "zone" {
  description = "The availability zone"
  default     = "europe-west2-c"
#  default = "us-west1-b"
}

variable "subnetwork" {
  description = "The subnetwork for the instance NIC"
  default     = "projects/clustered-cream/regions/europe-west2/subnetworks/default"
}

variable "sa_email" {
  description = "Email of the service account used to run the instance"
  default     = "637349472747-compute@developer.gserviceaccount.com"
}

variable "name" {
  description = "Name of the machine"
  default     = "ak-kraken-poc"
}

variable "AWS_ACCESS_KEY_ID" {
  sensitive   = true
  description = "AWS key for accessing S3"
}

variable "AWS_SECRET_ACCESS_KEY" {
  sensitive   = true
  description = "AWS Secret key for accessing S3"
}
variable "DAGSTER_PG_USERNAME" {
  sensitive   = true
  default     = "dagster"
  description = "Dagster Postgres db username"
}

variable "DAGSTER_PG_DB" {
  sensitive   = true
  default     = "ak-dagster-poc-db"
  description = "Dagster Postgres database name"
}

variable "NETWORK" {
  description = "Network for Compute to connect to internal services"
  default     = "projects/clustered-cream/global/networks/default"
}