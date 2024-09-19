module "gce-container" {
  source  = "terraform-google-modules/container-vm/google"
  version = "~> 2.0"

  container = {
    image = var.container_image
    env = [
      {
        name  = "AWS_ACCESS_KEY_ID"
        value = var.AWS_ACCESS_KEY_ID
      },
      {
        name  = "AWS_SECRET_ACCESS_KEY"
        value = var.AWS_SECRET_ACCESS_KEY
      },
      {
        name  = "DAGSTER_PG_USERNAME"
        value = google_sql_user.dagster.name
      },
      {
        name  = "DAGSTER_PG_PASSWORD"
        value = google_sql_user.dagster.password
      },
      {
        name  = "DAGSTER_PG_HOST"
        value = google_sql_database_instance.instance.private_ip_address
      },
      {
        name  = "DAGSTER_PG_DB"
        value = google_sql_database.database.name
      }
    ]

  }

  restart_policy = "Always"
}
