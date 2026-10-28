# RAG Evaluation Engine

Production GraphRAG pipeline and Deep Research Agent with Knowledge Graph augmentation, powered by **AWS Bedrock Agent Core** and **AWS Bedrock Foundation Models**.

---

## Architecture Overview

```
                                      +------------------------------------+
                                      |     Amazon Bedrock Agent Core      |
                                      |   (Claude 3.5 Sonnet / Haiku /     |
                                      |          Amazon Nova Pro)          |
                                      +-----------------+------------------+
                                                        |
                                           Action Group | OpenAPI 3.0
                                                        v
                                      +------------------------------------+
                                      | AWS Lambda Action Group Handler   |
                                      | (src.aws.lambda_action_group)      |
                                      +--------+------------------+--------+
                                               |                  |
                         +---------------------+                  +---------------------+
                         v                                                              v
      +-------------------------------------+                        +-------------------------------------+
      |         PostgreSQL + pgvector       |                        |       Neo4j Knowledge Graph         |
      |   - Dense Cosine Similarity Search  |                        |   - Ontology Multi-Hop Traversal    |
      |   - Sparse Keyword Search (BM25)    |                        |   - Concept & Entity Relationships  |
      +-------------------------------------+                        +-------------------------------------+
```

### Key Components

1. **AWS Bedrock Agent Core (`bedrock-agent` & `bedrock-agent-runtime`)**:
   - Managed cloud agent orchestration executing ReAct reasoning, step-by-step query decomposition, and context synthesis.
   - Grounded citations and attribution tracking natively returned to clients.
2. **Bedrock Foundation Models (`bedrock-runtime`)**:
   - High-throughput, serverless LLM generation via Anthropic Claude 3.5 Sonnet (`anthropic.claude-3-5-sonnet-20241022-v2:0`), Claude 3 Haiku, or Amazon Nova Pro.
   - Structured JSON schema enforcement conforming to `GroundedRAGResponse`.
3. **GraphRAG Action Group (`src/aws/lambda_action_group.py`)**:
   - Exposes tools to Bedrock Agent via OpenAPI 3.0 specification (`src/aws/action_group_openapi.json`).
   - Tools include `retrieve_and_rerank`, `hybrid_search`, `neo4j_traverse`, and `sanitize_input`.
4. **Hybrid Retrieval & Knowledge Graph Traversal**:
   - PostgreSQL (`pgvector` dense search + sparse keyword search).
   - Neo4j Cypher multi-hop graph traversal for interconnected technical entities.
   - Cross-Encoder reranking (`BAAI/bge-reranker-large`).

---

## Configuration (`.env`)

Configure AWS Bedrock and Bedrock Agent Core in your `.env` file:

```bash
# AWS Region & Credentials
AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your-key-id
# AWS_SECRET_ACCESS_KEY=your-secret-key

# LLM Provider: "bedrock" (managed cloud inference) or "local" (Qwen + LoRA)
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_TEMPERATURE=0.1
BEDROCK_MAX_TOKENS=1024

# Agent Orchestration Mode: "bedrock_agent" (AWS Bedrock Agent Core) or "orchestrator"
AGENT_MODE=bedrock_agent
BEDROCK_AGENT_ID=your-agent-id
BEDROCK_AGENT_ALIAS_ID=TSTALIASID  # Or "production"

# Databases
DATABASE_URL=postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=localdevpassword
```

---

## Deploying to AWS Bedrock Agent Core

### Option 1: Automated Deployment CLI (Recommended)

Run the deployment script to automatically create/update IAM roles, deploy the Lambda Action Group, configure Bedrock Agent Core, and set up a production alias:

```bash
python scripts/deploy_bedrock_agent.py \
  --region us-east-1 \
  --model-id "anthropic.claude-3-5-sonnet-20241022-v2:0" \
  --agent-name "rag-graph-research-agent" \
  --alias-name "production"
```

The script outputs your `BEDROCK_AGENT_ID` and `BEDROCK_AGENT_ALIAS_ID` upon completion.

### Option 2: AWS CloudFormation

Deploy the CloudFormation template:

```bash
aws cloudformation deploy \
  --template-file aws/bedrock_agent_stack.yaml \
  --stack-name production-graphrag-agent \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      FoundationModelId=anthropic.claude-3-5-sonnet-20241022-v2:0 \
      AgentName=production-graphrag-agent \
      DatabaseUrl="$DATABASE_URL" \
      Neo4jUri="$NEO4J_URI" \
  --region us-east-1
```

### Option 3: Terraform (IaC)

Deploy using the provided Terraform modules:

```bash
cd terraform
terraform init
terraform apply -var="aws_region=us-east-1"
```

### Option 4: CI/CD via GitHub Actions

The repository includes `.github/workflows/deploy.yml` which automatically:
1. Authenticates to AWS via GitHub OIDC role `GitHubActionsGraphRAGDeployRole`.
2. Builds and pushes the container image to Amazon ECR `graph-rag-api`.
3. Runs `scripts/deploy_bedrock_agent.py` to sync the Bedrock Agent and Action Group.
4. Updates the container service on ECS.

---

## Running the API

### Local / Docker Run

```bash
docker compose --profile api up --build -d
docker compose logs -f api
```

### API Endpoints

- **`POST /query`**: Executes orchestrated GraphRAG query with Bedrock foundation model generation.
- **`POST /bedrock-agent/query`**: Directly invokes AWS Bedrock Agent Core in the cloud:
  ```json
  {
    "query": "Explain how multi-head attention functions in transformer models."
  }
  ```
- **`GET /health`**: Returns system health including database, LangSmith tracing, and Bedrock configuration status.

---

## Running the Agent via Python

```python
from src.main_agent import DeepResearchAgent
from src.aws.bedrock_agent_runtime import BedrockAgentRunner

# Initialize agent with AWS Bedrock Agent Core
agent = DeepResearchAgent()
response = agent.run_research("Explain multi-head attention and transformer blocks.")

print(response.model_dump_json(indent=2))
```
