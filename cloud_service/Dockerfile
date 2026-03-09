FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

# --- Layer 1: dependency install (cached unless lock/pyproject change) ---
COPY uv.lock pyproject.toml ./
COPY cloud_service/pyproject.toml cloud_service/pyproject.toml
# Stub all workspace packages so uv can resolve deps without full source
COPY mujoco_sim/pyproject.toml mujoco_sim/pyproject.toml
RUN mkdir -p halo && touch halo/__init__.py && \
    mkdir -p cloud_service/cloud_service && touch cloud_service/cloud_service/__init__.py && \
    mkdir -p mujoco_sim/mujoco_sim && touch mujoco_sim/mujoco_sim/__init__.py
RUN uv sync --project cloud_service --no-dev --no-editable

# --- Layer 2: source code + configs (small, changes often) ---
COPY halo/ halo/
COPY cloud_service/ cloud_service/
COPY configs/ configs/
# Re-sync to install workspace packages in editable/linked mode
RUN uv sync --project cloud_service --no-dev

EXPOSE 8080

CMD ["uv", "run", "--project", "cloud_service", "uvicorn", "cloud_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
