

resource "google_sql_database" "database" {
  name     = var.DAGSTER_PG_DB
  instance = google_sql_database_instance.instance.name
  project  = var.project_id
}

# See versions at https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_database_instance#database_version
resource "google_sql_database_instance" "instance" {
  name             = "${var.DAGSTER_PG_DB}-instance"
  region           = "europe-west2"
  database_version = "POSTGRES_15"
  depends_on       = [google_service_networking_connection.private_vpc_connection]
  settings {
    tier              = "db-f1-micro"
    availability_type = "ZONAL"
    ip_configuration {
      private_network                               = var.NETWORK
      enable_private_path_for_google_cloud_services = true
    }
  }

  deletion_protection = "true"
}

resource "random_id" "dagster_password" {
  byte_length = 16
}

resource "google_sql_user" "dagster" {
  name     = var.DAGSTER_PG_USERNAME
  instance = google_sql_database_instance.instance.name
  password = random_id.dagster_password.id
}

resource "google_compute_global_address" "psql_internal_ip" {
  provider = google-beta

  name          = "psql-internal-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = var.NETWORK
  project       = var.project_id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  provider = google-beta

  network                 = var.NETWORK
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.psql_internal_ip.name]
}