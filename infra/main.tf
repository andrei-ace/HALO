terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# NOTE: APIs (run, artifactregistry, firestore, secretmanager, iam) must be
# enabled before the first apply — see cloud_service/README.md for the
# gcloud services enable commands.

# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "halo" {
  location      = var.region
  repository_id = "halo"
  format        = "DOCKER"
  description   = "HALO cognitive service container images"

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "2592000s" # 30 days
    }
  }

}

# ---------------------------------------------------------------------------
# Secret Manager
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "google-api-key"

  replication {
    auto {}
  }

}

# ---------------------------------------------------------------------------
# Firestore
# ---------------------------------------------------------------------------

resource "google_firestore_database" "default" {
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

}

# ---------------------------------------------------------------------------
# Service Account — Cloud Run runtime identity
# ---------------------------------------------------------------------------

resource "google_service_account" "cognitive" {
  account_id   = "halo-cognitive"
  display_name = "HALO Cognitive Service"
}

resource "google_project_iam_member" "cognitive_datastore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.cognitive.email}"
}

resource "google_secret_manager_secret_iam_member" "cognitive_secret_access" {
  secret_id = google_secret_manager_secret.google_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cognitive.email}"
}

# ---------------------------------------------------------------------------
# Cloud Run v2 Service
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "cognitive" {
  count    = var.deploy_service ? 1 : 0
  name     = "halo-cognitive"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cognitive.email

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    max_instance_request_concurrency = 10

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/halo/halo-cognitive:latest"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "HALO_PLANNER_MODEL"
        value = var.planner_model
      }
      env {
        name  = "HALO_VLM_MODEL"
        value = var.vlm_model
      }
      env {
        name  = "HALO_LIVE_AGENT_MODEL"
        value = var.live_agent_model
      }
      env {
        name  = "HALO_FIRESTORE_ENABLED"
        value = "true"
      }
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.google_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_artifact_registry_repository.halo,
    google_secret_manager_secret.google_api_key,
  ]
}

# ---------------------------------------------------------------------------
# Invoker Service Account — identity TUI clients use
# ---------------------------------------------------------------------------

resource "google_service_account" "invoker" {
  account_id   = "halo-tui-invoker"
  display_name = "HALO TUI Invoker"
}

resource "google_service_account_iam_member" "invoker_token_creator" {
  for_each           = toset(var.invoker_impersonators)
  service_account_id = google_service_account.invoker.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = each.value
}

resource "google_cloud_run_v2_service_iam_member" "invoker" {
  count    = var.deploy_service ? 1 : 0
  name     = google_cloud_run_v2_service.cognitive[0].name
  location = google_cloud_run_v2_service.cognitive[0].location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.invoker.email}"
}
