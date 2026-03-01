.PHONY: install install-sim test test-sim test-unit test-v test-file test-k test-component test-system test-e2e test-e2e-all test-integration generate-episodes generate-episodes-video visualize-ik tui-mock tui-live-videoloop tui-live-mujoco run-headless-mock run-headless-live sim-server ruff help

install:
	uv sync --extra dev

install-sim:
	uv sync --extra dev --extra sim

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

sim-server:
	uv run python -m mujoco_sim.server -v

ruff:
	uv run ruff check --fix .
	uv run ruff format .

EPISODES     ?= 1
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

IK_SEED    ?= 7
IK_OUT_DIR ?= data/ik_poses

visualize-ik:
	uv run python -m mujoco_sim.scripts.visualize_ik_pose \
		--seed $(IK_SEED) \
		--output-dir $(IK_OUT_DIR) \
		$(VIZ_ARGS)

test-integration:
	$(eval RUN_DIR := integration/runs/$(shell date +%Y%m%d_%H%M%S))
	mkdir -p $(RUN_DIR)
	uv run pytest integration/ -v -s \
		--tb=short \
		--junit-xml=$(RUN_DIR)/results.xml \
		2>&1 | tee $(RUN_DIR)/output.log

help:
	@echo "install            install deps (uv sync --extra dev)"
	@echo "install-sim        install deps + MuJoCo (uv sync --extra dev --extra sim)"
	@echo "generate-episodes        generate teacher episodes w/ VLM tracking (requires Ollama)"
	@echo "generate-episodes-video  same + save mp4 preview per episode (requires opencv)"
	@echo "test-sim           run mujoco_sim tests (requires --extra sim)"
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
	@echo "sim-server         start the MuJoCo sim ZMQ server (requires --extra sim)"
	@echo "visualize-ik       render IK-solved poses as PNGs (IK_SEED=7 IK_OUT_DIR=data/ik_poses)"
	@echo "tui-live-mujoco    launch the TUI with MuJoCo scene camera (requires Ollama + MuJoCo)"
