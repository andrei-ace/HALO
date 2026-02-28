.PHONY: install test test-unit test-v test-file test-k test-component test-system test-e2e test-e2e-all test-integration tui-mock tui-live-videoloop tui-live-mujoco run-headless-mock run-headless-live ruff help

install:
	uv sync --extra dev

test:
	uv run pytest

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

tui-live-videoloop:
	uv run python -m halo.tui.app --live \
		--arm $(ARM_ID) \
		--model $(PLANNER_MODEL) \
		--vlm-model $(VLM_MODEL) \
		--base-url $(OLLAMA_URL) \
		--source videoloop

tui-live-mujoco:
	uv run python -m halo.tui.app --live \
		--arm $(ARM_ID) \
		--model $(PLANNER_MODEL) \
		--vlm-model $(VLM_MODEL) \
		--base-url $(OLLAMA_URL) \
		--source mujoco

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

ruff:
	uv run ruff check --fix .
	uv run ruff format .

test-integration:
	$(eval RUN_DIR := integration/runs/$(shell date +%Y%m%d_%H%M%S))
	mkdir -p $(RUN_DIR)
	uv run pytest integration/ -v -s \
		--tb=short \
		--junit-xml=$(RUN_DIR)/results.xml \
		2>&1 | tee $(RUN_DIR)/output.log

help:
	@echo "install            install deps (uv sync --extra dev)"
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
	@echo "tui-live-videoloop launch the TUI with video loop source (requires Ollama)"
	@echo "tui-live-mujoco    launch the TUI with MuJoCo scene camera (requires Ollama + robosuite)"
