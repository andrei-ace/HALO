# Infrastructure

Terraform configuration for HALO's GCP infrastructure (Artifact Registry, Cloud Run, Firestore, Secret Manager, IAM).

## Files

| File | Purpose |
|------|---------|
| `main.tf` | All resource definitions (registry, secret, Firestore, service accounts, Cloud Run, IAM) |
| `variables.tf` | Input variables (project, region, models, invoker list, bootstrap flag) |
| `outputs.tf` | Outputs (project_id, service_url, artifact_registry path, invoker SA email) |
| `terraform.tfvars.example` | Example variable values — copy to `terraform.tfvars` |

## Resources

| Resource | Name | Purpose |
|----------|------|---------|
| `google_artifact_registry_repository` | `halo` | Docker image registry (30-day untagged cleanup) |
| `google_secret_manager_secret` | `google-api-key` | GOOGLE_API_KEY secret (auto-replicated) |
| `google_firestore_database` | `(default)` | Session/state persistence for cognitive service |
| `google_service_account` | `halo-cognitive` | Cloud Run runtime identity (Firestore + Secret access) |
| `google_cloud_run_v2_service` | `halo-cognitive` | Cognitive service (scale 0-1, 2 CPU / 2 Gi, concurrency 10) |
| `google_service_account` | `halo-tui-invoker` | TUI client identity for authenticated Cloud Run invocation |
| IAM bindings | — | `datastore.user` for cognitive SA, `secretAccessor` for API key, `run.invoker` for TUI SA, `serviceAccountTokenCreator` for impersonators |

## Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | — (required) | GCP project ID |
| `region` | string | `us-central1` | GCP region for all resources |
| `planner_model` | string | `gemini-3.1-flash-lite-preview` | Gemini model for planner |
| `vlm_model` | string | `gemini-3.1-flash-lite-preview` | Gemini model for VLM scene analysis |
| `live_agent_model` | string | `gemini-2.5-flash-native-audio-preview-12-2025` | Gemini model for Live Agent |
| `invoker_impersonators` | list(string) | `[]` | IAM members allowed to impersonate invoker SA |
| `deploy_service` | bool | `true` | `false` for bootstrap (registry + secrets only) |

## Outputs

| Output | Description |
|--------|-------------|
| `project_id` | GCP project ID |
| `service_url` | Cloud Run URL (empty until `deploy_service=true`) |
| `artifact_registry` | Full registry path for `docker push` |
| `invoker_sa_email` | TUI invoker service account email |

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make tf-init` | `terraform init` |
| `make tf-plan` | `terraform plan` |
| `make tf-apply` | `terraform apply` |
| `make tf-bootstrap` | Apply with `deploy_service=false` (registry + secrets only) |
| `make docker-push` | Build + push cognitive image to Artifact Registry |
| `make deploy-cloud` | `docker-push` then `tf-apply` with `deploy_service=true` |

## Key Design Notes

- **Bootstrap mode**: `deploy_service=false` creates registry + secrets first, so you can push an image before the Cloud Run service references it.
- **Dual service accounts**: `halo-cognitive` (runtime identity with Firestore + Secret access) and `halo-tui-invoker` (TUI client identity with `run.invoker` role).
- **Secret handling**: Secret shell is managed by Terraform; the secret *value* is set manually via `gcloud` (never in state).
- **Scale-to-zero**: `min_instance_count = 0` with `cpu_idle = true` and `startup_cpu_boost = true`.
- **GCP APIs**: Must be enabled before first apply — see `cloud_service/README.md` for the `gcloud services enable` commands.
- **Full deployment walkthrough**: See `cloud_service/README.md`.
