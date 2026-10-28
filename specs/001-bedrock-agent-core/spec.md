# Feature Specification: AWS Bedrock Agent Core & Bedrock LLM Integration

> **Status**: Implemented  
> **Author**: DeepMind Advanced Agentic Coding  
> **Date**: 2026-10-28  
> **Spec ID**: `specs/001-bedrock-agent-core`  
> **Related Files**: [`src/aws/`](file:///Users/sabyasachi/rag-evaluation-engine/src/aws/), [`tests/test_bedrock_integration.py`](file:///Users/sabyasachi/rag-evaluation-engine/tests/test_bedrock_integration.py)

---

## 1. Problem Statement & Context
- **Background**: The existing system relied on a local Qwen 1.5B model with LoRA adapters running on CPU/GPU, requiring heavy container images, model weight downloads, and significant compute overhead.
- **Objective**: Deploy and run the deep research agent on AWS using **AWS Bedrock Agent Core** (`bedrock-agent-runtime`) for autonomous multi-step reasoning, and **AWS Bedrock Foundation Models** (Claude 3.5 Sonnet / Haiku / Nova) for serverless generation, while exposing the hybrid GraphRAG retriever as an Action Group tool.
- **Out of Scope**: Migrating PostgreSQL pgvector or Neo4j databases to managed services (connections remain via VPC / connection URLs).

---

## 2. User Stories & Acceptance Criteria

### User Stories
- **US-1**: As an engineer, I want the agent to use AWS Bedrock Agent Core to orchestrate multi-step research and invoke hybrid retrieval tools automatically in the cloud.
- **US-2**: As an API consumer, I want to query the agent via `POST /bedrock-agent/query` and receive grounded JSON responses matching `GroundedRAGResponse`.
- **US-3**: As a DevOps engineer, I want an automated deployment script and CloudFormation/Terraform IaC to provision the agent, IAM roles, and Lambda action group.

### Acceptance Criteria
```gherkin
Scenario: Query AWS Bedrock Agent Core successfully
  Given a user question "Explain multi-head attention"
  When POST /bedrock-agent/query is called with the query
  Then the Bedrock Agent invokes the GraphRAG action group tools
  And streams back completion chunks and native attributions
  And returns a 200 OK with GroundedRAGResponse (answer, citations, confidence_score)

Scenario: Zero context fallback handling
  Given a query with no relevant documents
  When Bedrock generator or agent executes
  Then the response states no sufficiently relevant context was found
  And citations list is empty with confidence_score 0.0
```

---

## 3. Data Contracts & Interfaces

### Pydantic Schemas
```python
class Citation(BaseModel):
    chunk_id: str = Field(description="Evidence chunk identifier")
    source_doc: str = Field(description="Source document path or URI")
    relevant_snippet: str = Field(description="Direct text quote or paraphrased fact")

class GroundedRAGResponse(BaseModel):
    answer: str = Field(description="Precise technical answer grounded exclusively in context")
    citations: List[Citation] = Field(description="Exact chunk citations")
    confidence_score: float = Field(ge=0.0, le=1.0)
```

### Endpoints
| Method | Path | Request Body | Response Body | Status |
|---|---|---|---|---|
| `POST` | `/bedrock-agent/query` | `{"query": str, "session_id": Optional[str]}` | `GroundedRAGResponse` | 200, 500 |
| `GET` | `/health` | None | System status including Bedrock config | 200 |

---

## 4. Implementation Strategy

### Components
1. `src/aws/bedrock_generator.py`: Bedrock Converse API generator with JSON output schema enforcement.
2. `src/aws/lambda_action_group.py`: AWS Lambda handler responding to Bedrock Agent tool calls (`retrieve_and_rerank`, `hybrid_search`, `neo4j_traverse`, `sanitize_input`).
3. `src/aws/action_group_openapi.json`: OpenAPI 3.0 specification for tool parameters.
4. `src/aws/bedrock_agent_runtime.py`: Client for `bedrock-agent-runtime.invoke_agent`.
5. `scripts/deploy_bedrock_agent.py`: Turnkey deployment automation script.

---

## 5. Test Matrix & Verification

| Test Case | Scenario | Tested Component | Expected Result | Status |
|---|---|---|---|---|
| `TC-01` | Valid Converse API JSON | `BedrockGenerator` | Parses to `GroundedRAGResponse` | PASSED |
| `TC-02` | Markdown code block cleaning | `BedrockGenerator` | Strips ```` ```json ```` and parses | PASSED |
| `TC-03` | Empty retrieved chunks | `BedrockGenerator` | Returns fallback, skips API call | PASSED |
| `TC-04` | Event stream streaming | `BedrockAgentRunner` | Aggregates byte chunks into answer | PASSED |
| `TC-05` | Native Bedrock citations | `BedrockAgentRunner` | Maps `attribution` to `Citation` | PASSED |
| `TC-06` | Lambda Action Group routing | `lambda_action_group` | Returns Bedrock-compliant JSON response | PASSED |
| `TC-07` | Lambda function schema | `lambda_action_group` | Sanitizes query and redacts PII | PASSED |
| `TC-08` | Agent mode routing | `DeepResearchAgent` | Routes to Bedrock Runner when enabled | PASSED |

---

## 6. Observability & Guardrails
- **Input Guardrails**: `InputGuardrail.sanitize_and_redact()` runs on all queries to redact email addresses and phone numbers.
- **Output Guardrails**: Output parser verifies JSON structure and drops invalid citations if parsing fails.
- **Tracing**: Native Bedrock execution traces extracted from the event stream.

