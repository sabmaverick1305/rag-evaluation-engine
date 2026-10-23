FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HOME=/home/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --no-dev --no-install-project

# Copy application source code
COPY src ./src

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache
USER app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI via uvicorn
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
