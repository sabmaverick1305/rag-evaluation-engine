# RAG Evaluation Engine

## Run the API with Docker

```bash
docker compose --profile api up --build -d
docker compose logs -f api
```

Open <http://localhost:8000/docs> when Uvicorn reports startup is complete.
The API waits for PostgreSQL's health check, then loads the embedding model,
reranker, and Qwen generator. The first container start downloads model weights;
the `huggingface_cache` volume preserves them for subsequent starts. Docker on
macOS runs the models on Linux CPU and does not use Apple's Metal acceleration.

The Dockerfile uses Python 3.11 and installs production dependencies from
`uv.lock` with uv's `cpu` extra, then runs Uvicorn as a non-root user.
The optional extra selects CPU PyTorch builds on Linux following
[uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).
The Docker setup follows the
[uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/).
Build the image separately with `docker build -t rag-evaluation-engine .`.

Compose passes `.env` settings, including LangSmith configuration, to the API at
runtime. Credentials and local virtual environments are excluded from the image.
The API connects to the `postgres` service through Docker networking; Compose
overrides the host-based `DATABASE_URL` in `.env` accordingly. Use
`POSTGRES_PASSWORD` for a custom database password. See
[Compose environment configuration](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/).

The database must already contain your ingested document chunks. Initialization
SQL creates tables only when the database volume is first initialized; it does
not ingest PDFs or migrate existing tables. The existing `postgres_data` volume
is reused. Running `docker compose up -d postgres` continues to start only the
database; the API is enabled with the `api` profile.

Stop the services while retaining the database and model cache:

```bash
docker compose --profile api down
```
