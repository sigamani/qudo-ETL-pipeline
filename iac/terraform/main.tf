terraform {
  backend "gcs" {
    bucket = "ak-kraken-poc"
  }
}

resource "google_compute_address" "static" {
  name = "${var.name}-ipv4-address"
  region = substr(var.zone, 0, length(var.zone)-2)
}

resource "google_compute_firewall" "dagster" {
  project     = var.project_id
  name        = "allow-dagster"
  target_tags = ["dagster"]
  allow {
    protocol = "tcp"
    ports    = ["3000"]
  }
  network       = "default"
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_instance" "vm" {
  project      = var.project_id
  machine_type = "e2-standard-32"
  #machine_type = "c2d-highcpu-112"
  zone         = var.zone

  allow_stopping_for_update = true

  name = var.name

  boot_disk {
    initialize_params {
      image = module.gce-container.source_image
    }
  }

  network_interface {
    subnetwork = var.subnetwork
    access_config {
      nat_ip = google_compute_address.static.address
    }
  }

  tags = [
    "dagster"
  ]

  metadata = {
    gce-container-declaration = module.gce-container.metadata_value
    google-logging-enabled    = "true"
    google-monitoring-enabled = "true"
  }

  labels = {
    container-vm = module.gce-container.vm_container_label
  }

  service_account {
    email = var.sa_email
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}
