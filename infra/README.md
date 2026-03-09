# infra/

Terraform configuration that provisions all GCP infrastructure for the HALO cognitive service: container registry, secret management, database, Cloud Run service, and IAM.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GCP Project                                                │
│                                                             │
│  ┌─────────────────────┐    ┌────────────────────────────┐  │
│  │ Artifact Registry   │    │ Secret Manager             │  │
│  │ "halo" (Docker)     │───▶│ "google-api-key"           │  │
│  └─────────────────────┘    └────────────┬───────────────┘  │
│           │                              │ secretAccessor   │
│           │ image                        ▼                  │
│  ┌────────▼────────────────────────────────────────────┐    │
│  │ Cloud Run v2 "halo-cognitive"                       │    │
│  │   SA: halo-cognitive (datastore.user + secret)      │    │
│  │   Scale: 0–1, 2 CPU / 2 Gi, concurrency 10         │    │
│  └────────────────────────────────────┬────────────────┘    │
│                                       │                     │
│  ┌───────────────────┐                │ reads/writes        │
│  │ SA: halo-tui-     │  run.invoker   │                     │
│  │     invoker       │───────────────▶│                     │
│  └───────────────────┘                ▼                     │
│                              ┌─────────────────────┐        │
│                              │ Firestore (default)  │        │
│                              │ FIRESTORE_NATIVE     │        │
│                              └─────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Reference

### Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `project_id` | yes | — | GCP project ID |
| `region` | no | `us-central1` | GCP region |
| `planner_model` | no | `gemini-3.1-flash-lite-preview` | Planner model |
| `vlm_model` | no | `gemini-3.1-flash-lite-preview` | VLM model |
| `live_agent_model` | no | `gemini-2.5-flash-native-audio-preview-12-2025` | Live Agent model |
| `invoker_impersonators` | no | `[]` | IAM members that can impersonate the invoker SA |
| `deploy_service` | no | `true` | Set `false` for bootstrap (registry + secrets only) |

### Outputs

| Output | Description |
|--------|-------------|
| `project_id` | GCP project ID |
| `service_url` | Cloud Run URL (empty until service is deployed) |
| `artifact_registry` | Full registry path for `docker push` |
| `invoker_sa_email` | TUI invoker SA email |

## Deployment

For full deployment prerequisites (GCP API enablement, secret population, Docker auth) and step-by-step walkthrough, see [`cloud_service/README.md`](../cloud_service/README.md).

Quick start:

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
# Edit terraform.tfvars with your project_id

make tf-bootstrap    # Create registry + secrets (no Cloud Run yet)
make docker-push     # Build and push the image
# Populate the API key secret via gcloud (see cloud_service/README.md)
make deploy-cloud    # Push image + deploy Cloud Run service
```

## Provider Requirements

- Terraform >= 1.5
- Google provider ~> 6.0
