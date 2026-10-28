---
name: bedrock-agentcore
description: Operational guide for building, testing, and deploying GraphRAG agents on Amazon Bedrock AgentCore.
---

# Amazon Bedrock AgentCore Deployment & Operation Guide

This skill documents how to run, test, and deploy the autonomous GraphRAG Deep Research Agent on **Amazon Bedrock AgentCore** (`bedrock-agentcore`).

---

## 1. Why Amazon Bedrock AgentCore?

AWS placed **Bedrock Classic Agents** (`bedrock-agent create_agent`) in Maintenance Mode.
**Amazon Bedrock AgentCore** is AWS's modern, framework-agnostic platform built to support autonomous agent frameworks (**LangGraph**, **Pydantic AI**, **CrewAI**) in production.

### Core Architecture
- **Runtime Application (`src/aws/agentcore_app.py`)**: An ASGI application powered by `BedrockAgentCoreApp`. Exposes `@app.entrypoint` and `@app.ping`.
- **Autonomous Reasoning (`src/agent_workflow.py`)**: Multi-step ReAct agent with reflection loops, circuit breakers, and 4 tool interfaces (`retrieve_and_rerank`, `hybrid_search`, `neo4j_traverse`, `sanitize_input`).
- **Foundation Models (`src/aws/bedrock_generator.py`)**: Bedrock Converse API (Claude 3.5 Sonnet / Claude 3 Haiku / Amazon Nova Pro).
- **Client SDK (`src/aws/bedrock_agentcore_runtime.py`)**: Boto3 client (`boto3.client('bedrock-agentcore')`) invoking `invoke_agent_runtime`.

---

## 2. Local Development & Verification

### Running the AgentCore Server Locally
Start the AgentCore server with uvicorn on port `8080`:

```bash
# Using python directly
python3 src/aws/agentcore_app.py

# Or using the agentcore CLI
agentcore dev
```

### Health Ping Check
```bash
curl http://localhost:8080/ping
```
**Expected Response**:
```json
{"status": "HEALTHY"}
```

### Invoking the Agent Locally
```bash
# Direct AgentCore invocation
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain multi-head attention in transformers"}'

# Or via FastAPI service endpoint
curl -X POST http://localhost:8000/agentcore/research \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain multi-head attention in transformers"}'
```

---

## 3. Remote Invocation via AWS Python SDK

When deployed to an Amazon Bedrock AgentCore Runtime ARN:

```python
import os
from src.aws.bedrock_agentcore_runtime import BedrockAgentCoreRuntime

# Initialize runtime client
runtime = BedrockAgentCoreRuntime(
    runtime_arn="arn:aws:bedrock-agentcore:us-east-1:159142082968:runtime/rag-graph-research-agent",
    region_name="us-east-1"
)

# Invoke the agent
response = runtime.invoke_runtime(
    prompt="Explain how knowledge graphs enhance RAG retrieval precision."
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score}")
for citation in response.citations:
    print(f"- [{citation.chunk_id}] {citation.source_doc}: {citation.relevant_snippet}")
```

---

## 4. Verification & Quality Gates

Run the automated inspection gate before shipping:

```bash
python3 scripts/verify_and_review.py
```
This runs:
1. Python syntax & compilation (`py_compile`)
2. Ruff lint & quality standards
3. Hermetic unit tests (`tests/test_agentcore_integration.py`, `tests/test_agent_workflow.py`, `tests/test_bedrock_integration.py`)
