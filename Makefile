.PHONY: install test test-v test-file test-k test-integration tui help

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

tui:
	uv run python -m halo.tui.app

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
	@echo "tui               launch the HALO terminal dashboard"
