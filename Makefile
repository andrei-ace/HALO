.PHONY: install test test-v test-file test-k test-integration tui-mock tui-live help

install:
	uv sync --extra dev

test:
	uv run pytest

test-v:
	uv run pytest -v

test-file:
	uv run pytest $(FILE)

test-k:
	uv run pytest -k "$(K)"

tui-mock:
	uv run python -m halo.tui.app

tui-live:
	uv run --extra planner python -m halo.tui.app --live --arm arm0 --model gpt-oss --base-url http://localhost:11434

test-integration:
	$(eval RUN_DIR := integration/runs/$(shell date +%Y%m%d_%H%M%S))
	mkdir -p $(RUN_DIR)
	uv run pytest integration/ -v -s \
		--tb=short \
		--junit-xml=$(RUN_DIR)/results.xml \
		2>&1 | tee $(RUN_DIR)/output.log

help:
	@echo "install    install deps (uv sync --extra dev)"
	@echo "test       run all tests"
	@echo "test-v     run all tests (verbose)"
	@echo "test-file  run one file:  make test-file FILE=tests/test_foo.py"
	@echo "test-k     run by name:   make test-k K=test_snapshot_ids_increment"
	@echo "test-integration  run LLM integration tests (requires Ollama)"
	@echo "tui-mock          launch the HALO terminal dashboard (mock mode, no Ollama needed)"
	@echo "tui-live          launch the TUI wired to HALORuntime + PlannerAgent (requires Ollama)"
