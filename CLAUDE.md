# CLAUDE.md - Repository Standards & Governance

## Project Overview
`rag-evaluation-engine` is a production GraphRAG and Deep Research Agent system combining:
- **Dense & Sparse Retrieval**: PostgreSQL with `pgvector` (cosine distance) and keyword search.
- **Knowledge Graph Augmentation**: Neo4j multi-hop ontology traversal for entity and concept relationships.
- **Reranking**: Cross-Encoder (`BAAI/bge-reranker-large`).
- **Generation & Orchestration**: AWS Bedrock Foundation Models (Claude 3.5 Sonnet / Haiku / Nova) and **AWS Bedrock Agent Core** (`bedrock-agent` / `bedrock-agent-runtime`), with local Qwen fallback.
- **Serving & Observability**: FastAPI REST API with LangSmith distributed tracing.

---

## Spec-Driven Development (SDD) Workflow
All non-trivial feature additions and architectural modifications follow Spec-Driven Development:
1. **Spec First (`specs/<feature-name>/spec.md`)**:
   - Create or update the feature spec using `specs/template.md`.
   - Define user stories, Pydantic data contracts, and the test matrix *before* implementing code.
2. **Test Driven Verification (`tests/test_<feature>.py`)**:
   - Write unit tests and fixtures matching the spec's test criteria.
   - Run tests to confirm expected failure prior to implementation.
3. **Implementation (`src/`)**:
   - Implement minimal, robust code to satisfy the spec and pass all tests.
4. **Procedural Skill Capture (`skills/<workflow-name>/SKILL.md`)**:
   - When a repeatable operational task is established (e.g., deployments, benchmarks), capture the recipe in a `SKILL.md`.

---

## Repository Structure
```
rag-evaluation-engine/
├── CLAUDE.md                      # Global repository governance and conventions
├── README.md                      # User-facing onboarding and architecture guide
├── specs/                         # Feature specifications & test fixtures
│   ├── template.md                # Standard spec template
│   └── <feature-id>/spec.md       # Specific feature design and test matrix
├── skills/                        # Reusable operational workflow guides
├── src/                           # Application source code
│   ├── api.py                     # FastAPI application endpoints
│   ├── main_agent.py              # DeepResearchAgent pipeline
│   ├── orchestrator.py            # GraphRAG chaining orchestrator
│   ├── retrieval.py               # Hybrid pgvector + BM25 retriever
│   ├── neo4j_client.py            # Neo4j Cypher ontology traversal client
│   ├── generator.py               # Local HuggingFace/LoRA generator
│   ├── guardrails.py              # PII redaction and output verification
│   ├── schemas.py                 # Core Pydantic data schemas
│   ├── state.py                   # RAG state tracking model
│   └── aws/                       # AWS Bedrock & Bedrock Agent Core integration
│       ├── bedrock_generator.py   # Bedrock Converse API LLM generator
│       ├── bedrock_agent_runtime.py # Bedrock Agent Core invoker
│       ├── lambda_action_group.py # Lambda handler for Bedrock Agent tools
│       └── action_group_openapi.json # OpenAPI 3.0 tool definitions
├── scripts/                       # Automation and deployment CLI scripts
├── aws/                           # CloudFormation infrastructure templates
├── terraform/                     # Terraform IaC configurations
└── tests/                         # Pytest test suite
```

---

## Coding Standards & Conventions

### Language & Formatting
- **Python Version**: Python 3.11+ compatibility.
- **Type Annotations**: Explicit type hints are mandatory on all public function signatures, parameters, and return types.
- **Data Models**: Use **Pydantic v2** (`BaseModel`, `Field`) for all request/response contracts, API payloads, and internal state representations. Avoid unstructured dictionaries for external contracts.
- **Naming Conventions**:
  - Modules/Files: `snake_case.py`
  - Classes: `PascalCase`
  - Functions & Variables: `snake_case`
  - Constants & Environment Variables: `UPPER_SNAKE_CASE`
  - Test files: `tests/test_<module_name>.py`

### Safety, Guardrails & Fallbacks
- **Resilient Imports**: Database and cloud drivers (`boto3`, `neo4j`, `psycopg2`) must be imported safely or lazily loaded in handlers to prevent unnecessary cold start penalties and test suite breakage.
- **Grounded Output Schema**: LLM generation must conform to `GroundedRAGResponse` (`answer`, `citations`, `confidence_score`). Never fabricate citations when retrieval returns no relevant context.
- **Sanitization**: All user queries must pass through `InputGuardrail.sanitize_and_redact()` prior to database querying or external LLM transmission.

---

## Testing & Verification Guidelines

1. **Isolation & Fast Feedback**:
   - Unit tests must be hermetic and execute in < 2 seconds.
   - Do **NOT** make live network or database calls in unit tests.
   - Mock all external dependencies: AWS Bedrock (`boto3`), PostgreSQL (`psycopg2`), Neo4j (`GraphDatabase`), and LangSmith.
2. **Fixture Discipline**:
   - Store complex test inputs/outputs as structured fixtures in `specs/<feature>/fixtures/` or `conftest.py`.
3. **Async / Sync Contracts**:
   - Verify both synchronous execution and async streaming response handling where applicable.
4. **Coverage Targets**:
   - High test coverage for schema validation, error boundaries, LLM response parsing (e.g. malformed JSON, code fence stripping), and fallback branches.

---

## Common Development Commands

### Environment & Dependencies
```bash
# Sync dependencies using uv
uv sync --extra cpu
```

### Running Tests
```bash
# Run entire test suite
pytest

# Run specific test file with verbose output
pytest tests/test_bedrock_integration.py -v

# Run with coverage report
pytest --cov=src tests/
```

### Running the API
```bash
# Run local FastAPI server
uv run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Run with Docker Compose
docker compose --profile api up --build -d
```

### Deploying AWS Bedrock Agent Core
```bash
# Deploy / update Bedrock Agent & Lambda Action Group
python scripts/deploy_bedrock_agent.py --region us-east-1 --model-id "anthropic.claude-3-5-sonnet-20241022-v2:0"
```

