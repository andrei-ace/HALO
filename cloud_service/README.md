# HALO Cloud Cognitive Service

FastAPI service that runs the HALO planner (ADK ReAct agent) and VLM (Gemini) on GCP Cloud Run. The robot host connects as a thin HTTP client via `RemoteCognitiveBackend`.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/decide` | POST | Planner decision — snapshot JSON in, commands JSON out |
| `/vlm/scene` | POST | VLM scene analysis — JPEG image + metadata in, VlmScene JSON out |
| `/state/{arm_id}` | GET | Session readiness and cursor for a specific arm |
| `/health` | GET | Health check |
| `/ws/live/{arm_id}` | WS | Live Agent bidirectional audio+text streaming |

## Quick start: local testing against Gemini

1. **Get a Gemini API key** at https://aistudio.google.com/apikey

2. **Create your `.env` file:**
   ```bash
   cp cloud_service/.env.example cloud_service/.env
   # Edit .env and paste your GOOGLE_API_KEY
   ```

3. **One-command smoke test** (verifies health, decide, state):
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

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | Yes | — | Gemini API key |
| `HALO_PLANNER_MODEL` | No | `gemini-3.1-flash-lite-preview` | Planner model |
| `HALO_VLM_MODEL` | No | `gemini-3.1-flash-lite-preview` | VLM model |
| `HALO_COMPACTION_INTERVAL` | No | `20` | Invocations between context compaction |
| `HALO_COMPACTION_OVERLAP` | No | `4` | Recent invocations kept uncompacted |
| `HALO_FIRESTORE_ENABLED` | No | auto-detect | Enable Firestore persistence (auto-enabled when `FIRESTORE_EMULATOR_HOST` is set) |
| `HALO_FIRESTORE_COLLECTION` | No | `halo_sessions` | Firestore collection name |
| `HALO_FIRESTORE_TTL_HOURS` | No | `1.0` | Session TTL in Firestore |
| `HALO_LIVE_AGENT_ENABLED` | No | `true` | Enable Live Agent WebSocket endpoint |
| `HALO_LIVE_AGENT_MODEL` | No | `gemini-2.5-flash-native-audio-preview-12-2025` | Gemini Live model |
| `HALO_LIVE_AGENT_VOICE` | No | `Kore` | Live Agent voice |

## Cloud Run deployment

Auth is handled at the Cloud Run IAM layer (not app-level Bearer tokens).
All GCP resources are managed via Terraform in `infra/`.

### Prerequisites

Terraform needs a GCP account with sufficient permissions. **Owner** covers
everything; otherwise you need at least:

- `roles/serviceusage.serviceUsageAdmin` (enable APIs)
- `roles/iam.serviceAccountAdmin` (create service accounts)
- `roles/artifactregistry.admin`
- `roles/secretmanager.admin`
- `roles/run.admin`
- `roles/datastore.owner` (Firestore)

Authenticate so Terraform can use your credentials:

```bash
gcloud auth application-default login
```

The project must have a billing account linked — APIs and Cloud Run won't work
without it:

```bash
# List available billing accounts
gcloud billing accounts list

# Link one to your project
gcloud billing projects link $PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX
```

Terraform manages API enablement automatically, but the **Service Usage API**
itself must be enabled first (bootstrap dependency). Enable it along with the
other required APIs manually before the first apply:

```bash
PROJECT_ID=your-project-id

gcloud services enable serviceusage.googleapis.com    --project=$PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com              --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com    --project=$PROJECT_ID
gcloud services enable firestore.googleapis.com        --project=$PROJECT_ID
gcloud services enable iam.googleapis.com              --project=$PROJECT_ID
```

### Steps

```bash
# 1. Initialise Terraform (one-time)
make tf-init

# 2. Create infrastructure (registry, secrets, SAs, Firestore — no Cloud Run yet)
make tf-bootstrap

# 3. Configure Docker to authenticate with Artifact Registry (one-time)
gcloud auth configure-docker $(cd infra && terraform output -raw artifact_registry | cut -d/ -f1)

# 4. Add your Gemini API key to Secret Manager (one-time)
echo -n "YOUR_GEMINI_KEY" | gcloud secrets versions add google-api-key \
  --project=$(cd infra && terraform output -raw project_id) --data-file=-

# 5. Build + push the image, then create the Cloud Run service
make deploy-cloud
```

Subsequent deploys only need `make deploy-cloud`.

> **Why is the API key manual?** Terraform creates the Secret Manager secret
> and IAM bindings, but the actual key value is added via `gcloud` to avoid
> storing it in Terraform state (which keeps secrets in plaintext). To rotate
> the key, run the `gcloud secrets versions add` command again.

## TUI authentication

The TUI authenticates to Cloud Run using GCP identity tokens.
It impersonates the invoker service account to mint tokens:

```bash
CLOUD_URL=$(cd infra && terraform output -raw service_url)
SA_EMAIL=$(cd infra && terraform output -raw invoker_sa_email)

gcloud auth application-default login
make tui-live-cloud HALO_CLOUD_URL=$CLOUD_URL SA_EMAIL=$SA_EMAIL
```

The `--sa-email` flag tells the TUI to impersonate the invoker SA using your
user ADC credentials.

### Granting impersonation rights

**All users** (including project Owners) must be listed in
`invoker_impersonators` to impersonate the invoker SA. Project-level Owner
grants broad permissions, but SA-level IAM bindings are managed as separate
Terraform resources (`google_service_account_iam_member`) and are not
inherited from project-level roles.

Add your email to `invoker_impersonators` in `infra/terraform.tfvars`:

```hcl
invoker_impersonators = ["user:you@example.com"]
```

Then apply:

```bash
cd infra && terraform apply
```

Without this, you will get a `Permission 'iam.serviceAccounts.getOpenIdToken'
denied` error when launching the TUI.

Alternatively, create a service account key file:

```bash
PROJECT=$(cd infra && terraform output -raw project_id)
SA_EMAIL=$(cd infra && terraform output -raw invoker_sa_email)
gcloud iam service-accounts keys create invoker-key.json \
  --iam-account="$SA_EMAIL" --project="$PROJECT"

uv run python -m halo.tui.app --live \
  --cloud-url $CLOUD_URL \
  --sa-key-file invoker-key.json \
  --source mujoco --live-agent
```

> **Note:** Key files (`*-key.json`, `*-credentials.json`) are gitignored. Never commit them.

### Local development (no auth)

When connecting to `http://localhost:8080`, IAM auth is automatically skipped:

```bash
make run-cloud-service    # terminal 1
make tui-live-cloud-local # terminal 2
```
