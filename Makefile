.PHONY: install install-sim test test-sim test-unit test-v test-file test-k test-component test-system test-e2e test-e2e-all test-integration test-cloud-service smoke-cloud-service test-cloud-service-integration generate-episodes generate-episodes-video generate-episodes-place visualize-ik tui-mock tui-live tui-live-cloud tui-live-cloud-local run-headless-mock run-headless-live run-cloud-service sim-server ruff tf-init tf-plan tf-apply tf-bootstrap docker-push deploy-cloud help

install:
	uv sync --extra dev --extra sim

install-sim: install

test:
	uv run pytest

test-sim:
	uv run pytest mujoco_sim/mujoco_sim/tests/ -v

test-unit:
	uv run pytest tests/ --ignore=tests/component --ignore=tests/system --ignore=tests/e2e

test-v:
	uv run pytest -v

test-file:
	uv run pytest $(FILE)

test-k:
	uv run pytest -k "$(K)"

tui-mock:
	uv run python -m halo.tui.app

PLANNER_MODEL  ?= gpt-oss:20B
VLM_MODEL      ?= qwen2.5vl:3b
OLLAMA_URL     ?= http://localhost:11434
ARM_ID         ?= arm0

tui-live:
	uv run python -m halo.tui.app --live \
		--arm $(ARM_ID) \
		--model $(PLANNER_MODEL) \
		--vlm-model $(VLM_MODEL) \
		--base-url $(OLLAMA_URL) \
		--source mujoco

HALO_CLOUD_URL ?= $(shell cd infra && terraform output -raw service_url 2>/dev/null)
SA_EMAIL       ?= $(shell cd infra && terraform output -raw invoker_sa_email 2>/dev/null)

tui-live-cloud:
	@if [ -z "$(HALO_CLOUD_URL)" ]; then echo "Error: HALO_CLOUD_URL is empty. Set it or run 'terraform apply' in infra/."; exit 1; fi
	uv run python -m halo.tui.app --live \
		--arm $(ARM_ID) \
		--cloud-url $(HALO_CLOUD_URL) \
		$(if $(SA_EMAIL),--sa-email $(SA_EMAIL)) \
		--source mujoco \
		--live-agent

CLOUD_PLANNER_MODEL    ?= gemini-3.1-flash-lite-preview
CLOUD_VLM_MODEL        ?= gemini-3.1-flash-lite-preview
COMPACTION_INTERVAL    ?= 8
COMPACTION_OVERLAP     ?= 4

tui-live-cloud-local:
	uv run python -m halo.tui.app --live \
		--arm $(ARM_ID) \
		--cloud-url http://localhost:8080 \
		--source mujoco \
		--live-agent

test-component:
	uv run pytest tests/component/ -v

test-system:
	uv run pytest tests/system/ -v

test-e2e:
	uv run pytest tests/e2e/ -v -s $(ARGS)

test-e2e-all:
	uv run pytest tests/e2e/ -v -s --run-all-vlm-models

run-headless-mock:
	uv run python -m halo.runner --mock --duration 30

run-headless-live:
	uv run python -m halo.runner \
		--model $(PLANNER_MODEL) \
		--vlm-model $(VLM_MODEL) \
		--base-url $(OLLAMA_URL) \
		--arm $(ARM_ID)

run-cloud-service:
	HALO_COMPACTION_INTERVAL=$(COMPACTION_INTERVAL) \
	HALO_COMPACTION_OVERLAP=$(COMPACTION_OVERLAP) \
	uv run --project cloud_service uvicorn cloud_service.app:app --host 0.0.0.0 --port 8080 --reload --reload-dir .

test-cloud-service:
	uv run pytest cloud_service/tests/ -v --ignore=cloud_service/tests/test_integration.py

smoke-cloud-service:
	uv run python cloud_service/scripts/smoke_test.py

test-cloud-service-integration:
	uv run pytest cloud_service/tests/test_integration.py -v -s

sim-server:
	uv run python -m mujoco_sim.server -v

ruff:
	uv run ruff check --fix .
	uv run ruff format .

EPISODES     ?= 16
EPISODE_DIR  ?= data/episodes
SEED_BASE    ?= 0

generate-episodes:
	uv run python -m mujoco_sim.scripts.generate_episodes \
		--num-episodes $(EPISODES) \
		--output-dir $(EPISODE_DIR) \
		--seed-base $(SEED_BASE) \
		$(GENERATE_ARGS)

generate-episodes-video:
	uv run python -m mujoco_sim.scripts.generate_episodes \
		--num-episodes $(EPISODES) \
		--output-dir $(EPISODE_DIR) \
		--seed-base $(SEED_BASE) \
		--save-video \
		$(GENERATE_ARGS)

generate-episodes-place:
	uv run python -m mujoco_sim.scripts.generate_episodes \
		--num-episodes $(EPISODES) \
		--output-dir $(EPISODE_DIR) \
		--seed-base $(SEED_BASE) \
		--save-video \
		--pick-and-place \
		$(GENERATE_ARGS)

IK_SEED    ?= 7
IK_OUT_DIR ?= data/ik_poses

visualize-ik:
	uv run python -m mujoco_sim.scripts.visualize_ik_pose \
		--seed $(IK_SEED) \
		--output-dir $(IK_OUT_DIR) \
		$(VIZ_ARGS)

# ---------------------------------------------------------------------------
# Terraform
# ---------------------------------------------------------------------------

tf-init:
	cd infra && terraform init

tf-plan:
	cd infra && terraform plan

tf-apply:
	cd infra && terraform apply

# Build + push Docker image to Artifact Registry.
# Falls back to constructing the path from GCP_PROJECT_ID / GCP_REGION env vars
# when terraform output is not yet available (e.g. registry created but no full apply).
GCP_PROJECT_ID ?=
GCP_REGION     ?=

docker-push:
	$(eval REPO := $(or \
		$(shell cd infra && terraform output -raw artifact_registry 2>/dev/null), \
		$(if $(GCP_PROJECT_ID),$(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/halo)))
	@test -n "$(REPO)" || { echo "ERROR: Set GCP_PROJECT_ID or run 'make tf-apply' first"; exit 1; }
	docker build --platform linux/amd64 -t $(REPO)/halo-cognitive:latest -f cloud_service/Dockerfile .
	docker push $(REPO)/halo-cognitive:latest

# Bootstrap (first-time only): create registry, secrets, SAs, Firestore — no Cloud Run yet.
# After this, push the image and add the API key secret, then run deploy-cloud.
tf-bootstrap:
	cd infra && terraform apply -var deploy_service=false

# Full deploy: push image then create/update the Cloud Run service.
# Requires: tf-bootstrap done, image pushable, API key secret populated.
deploy-cloud:
	$(MAKE) docker-push
	cd infra && terraform apply -var deploy_service=true

# ---------------------------------------------------------------------------

test-integration:
	$(eval RUN_DIR := integration/runs/$(shell date +%Y%m%d_%H%M%S))
	mkdir -p $(RUN_DIR)
	uv run pytest integration/ -v -s \
		--tb=short \
		--junit-xml=$(RUN_DIR)/results.xml \
		2>&1 | tee $(RUN_DIR)/output.log

help:
	@echo "install            install deps + MuJoCo (uv sync --extra dev)"
	@echo "install-sim        alias for install"
	@echo "generate-episodes        generate teacher episodes w/ VLM tracking (requires Ollama)"
	@echo "generate-episodes-video  same + save mp4 preview per episode (requires opencv)"
	@echo "generate-episodes-place  pick-and-place episodes + video + VLM tracking"
	@echo "test-sim           run mujoco_sim tests"
	@echo "test               run all unit tests"
	@echo "test-unit          run unit tests (excluding component/system/e2e)"
	@echo "test-v             run all tests (verbose)"
	@echo "test-file          run one file:  make test-file FILE=tests/test_foo.py"
	@echo "test-k             run by name:   make test-k K=test_snapshot_ids_increment"
	@echo "test-component     run component tests (mocked latency, no Ollama)"
	@echo "test-system        run system tests (all services, mocked deps)"
	@echo "test-e2e           run E2E tests (real Ollama + VLM, 3b only)"
	@echo "test-e2e-all       run E2E tests with all VLM models (3b + 7b)"
	@echo "test-integration   run LLM integration tests (requires Ollama)"
	@echo "ruff               run ruff check --fix + format"
	@echo "run-headless-mock  run headless HALO (mock mode)"
	@echo "run-headless-live  run headless HALO (live, requires Ollama)"
	@echo "tui-mock           launch the HALO terminal dashboard (mock mode)"
	@echo "sim-server         start the MuJoCo sim ZMQ server"
	@echo "visualize-ik       render IK-solved poses as PNGs (IK_SEED=7 IK_OUT_DIR=data/ik_poses)"
	@echo "tui-live           launch TUI with local Ollama + MuJoCo sim"
	@echo "tui-live-cloud     launch TUI via cloud service HTTP (set HALO_CLOUD_URL for remote)"
	@echo "tui-live-cloud-local  launch TUI against a locally running cloud service"
	@echo "run-cloud-service  run cloud_service backed by Gemini (requires GOOGLE_API_KEY)"
	@echo "test-cloud-service         run cloud_service unit tests (no API key needed)"
	@echo "smoke-cloud-service        one-command smoke test against Gemini (requires GOOGLE_API_KEY)"
	@echo "test-cloud-service-integration  run cloud_service integration tests (requires GOOGLE_API_KEY)"
