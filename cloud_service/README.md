# HALO Cloud Cognitive Service

FastAPI service that runs the HALO planner (ADK ReAct agent) and VLM (Gemini) on GCP Cloud Run. The robot host connects as a thin HTTP client via `CloudCognitiveBackend`.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | Planner decision — snapshot JSON in, commands JSON out |
| `/vlm/scene` | POST | VLM scene analysis — JPEG image + metadata in, VlmScene JSON out |
| `/warm-up` | POST | Warm up session with CognitiveState + journal entries |
| `/state/{arm_id}` | GET | Session readiness and cursor for a specific arm |
| `/health` | GET | Health check |

## Quick start: local testing against Gemini

1. **Get a Gemini API key** at https://aistudio.google.com/apikey

2. **Create your `.env` file:**
   ```bash
   cp cloud_service/.env.example cloud_service/.env
   # Edit .env and paste your GOOGLE_API_KEY
   ```

3. **One-command smoke test** (verifies health, warm-up, decide, state):
   ```bash
   GOOGLE_API_KEY=<your-key> make smoke-cloud-service
   ```

4. **Two-terminal workflow** (full TUI + cloud planner + MuJoCo sim):
   ```bash
   # Terminal 1: start the cloud service
   GOOGLE_API_KEY=<your-key> make run-cloud-service

   # Terminal 2: start the MuJoCo sim server
   make sim-server

   # Terminal 3: connect TUI to cloud service + sim
   make tui-live-cloud
   ```

5. **Test commands:**
   ```bash
   make test-cloud-service                              # unit tests (no key needed)
   GOOGLE_API_KEY=<key> make smoke-cloud-service         # smoke test
   GOOGLE_API_KEY=<key> make test-cloud-service-integration  # integration tests
   ```

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
| `HALO_PLANNER_MODEL` | No | Planner model (default: `gemini-3.1-flash-lite-preview`) |
| `HALO_VLM_MODEL` | No | VLM model (default: `gemini-3.1-flash-lite-preview`) |

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
