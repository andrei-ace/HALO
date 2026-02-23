.PHONY: install test test-v test-file test-k test-integration help

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

test-integration:
	uv run pytest integration/ -v -s

help:
	@echo "install    install deps (uv sync --extra dev)"
	@echo "test       run all tests"
	@echo "test-v     run all tests (verbose)"
	@echo "test-file  run one file:  make test-file FILE=tests/test_foo.py"
	@echo "test-k     run by name:   make test-k K=test_snapshot_ids_increment"
	@echo "test-integration  run LLM integration tests (requires Ollama)"
