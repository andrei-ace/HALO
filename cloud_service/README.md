# HALO Cloud Cognitive Service

FastAPI service that runs the HALO planner (ADK ReAct agent) and VLM (Gemini) on GCP Cloud Run. The robot host connects as a thin HTTP client via `CloudCognitiveBackend`.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | Planner decision — snapshot JSON in, commands JSON out |
| `/vlm/scene` | POST | VLM scene analysis — JPEG image + metadata in, VlmScene JSON out |
| `/health` | GET | Health check |
| `/reset` | POST | Reset planner session state |

## Local development

```bash
# From repo root
uv sync --project cloud_service --extra dev
uv run --project cloud_service uvicorn cloud_service.app:app --reload --port 8080

# Run tests
uv run --project cloud_service pytest cloud_service/tests/ -v
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `HALO_CLOUD_API_KEY` | No | Bearer token clients must present (skip auth if unset) |
| `HALO_PLANNER_MODEL` | No | Planner model (default: `gemini-2.5-flash`) |
| `HALO_VLM_MODEL` | No | VLM model (default: `gemini-2.5-flash`) |

## Cloud Run deployment

```bash
gcloud run deploy halo-cognitive \
    --source . \
    --memory 2Gi --cpu 2 \
    --min-instances 1 --max-instances 4 \
    --concurrency 1 \
    --timeout 60 \
    --set-secrets "GOOGLE_API_KEY=halo-google-api-key:latest,HALO_CLOUD_API_KEY=halo-cloud-api-key:latest"
```
