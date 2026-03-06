FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system deps for opencv (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace (halo + cloud_service)
COPY . .

# Install cloud_service and all its deps
RUN uv sync --project cloud_service

EXPOSE 8080

CMD ["uv", "run", "--project", "cloud_service", "uvicorn", "cloud_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
