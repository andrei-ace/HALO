output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "service_url" {
  description = "Cloud Run service URL (empty until deploy_service=true)"
  value       = var.deploy_service ? google_cloud_run_v2_service.cognitive[0].uri : ""
}

output "artifact_registry" {
  description = "Full Artifact Registry path for docker push"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.halo.repository_id}"
}

output "invoker_sa_email" {
  description = "TUI invoker service account email (create a key for local dev)"
  value       = google_service_account.invoker.email
}
