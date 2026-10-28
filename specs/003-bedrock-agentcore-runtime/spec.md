# Feature Specification: Amazon Bedrock AgentCore Runtime Integration

> **Status**: In Progress  
> **Author**: DeepMind Advanced Agentic Coding  
> **Date**: 2026-10-28  
> **Spec ID**: `specs/003-bedrock-agentcore-runtime`  
> **Related Files**: [`src/aws/agentcore_app.py`](file:///Users/sabyasachi/rag-evaluation-engine/src/aws/agentcore_app.py), [`src/aws/bedrock_agentcore_runtime.py`](file:///Users/sabyasachi/rag-evaluation-engine/src/aws/bedrock_agentcore_runtime.py), [`src/main_agent.py`](file:///Users/sabyasachi/rag-evaluation-engine/src/main_agent.py), [`src/api.py`](file:///Users/sabyasachi/rag-evaluation-engine/src/api.py)

---

## 1. Problem Statement & Context
- **Background**: AWS has placed "Bedrock Agents Classic" (`bedrock-agent create_agent`) into Maintenance Mode and closed new agent creation for accounts without prior classic usage history.
- **Modern Solution**: AWS has officially introduced **Amazon Bedrock AgentCore** (`bedrock-agentcore`), a modular, framework-agnostic runtime and platform designed to securely run autonomous agent workflows (such as **LangGraph**, **Pydantic AI**, and **CrewAI**) in production.
- **Objective**: Move our production GraphRAG research agent to **Amazon Bedrock AgentCore**:
  1. Wrap the autonomous LangGraph workflow (`src/agent_workflow.py`) in an official `BedrockAgentCoreApp` server.
  2. Implement `@app.entrypoint` to process invocation requests with PII redaction and `GroundedRAGResponse` enforcement.
  3. Implement `@app.ping` health checks.
  4. Build `BedrockAgentCoreRuntime` client supporting `boto3.client('bedrock-agentcore')` (`invoke_agent_runtime`).
  5. Add `DeepResearchAgent(mode="agentcore")` and `POST /agentcore/research` in the FastAPI surface.

---

## 2. User Stories & Acceptance Criteria

### User Stories
- **US-1**: As an engineer, I want the autonomous LangGraph agent to run on the Amazon Bedrock AgentCore runtime (`BedrockAgentCoreApp`) so that it deploys natively to AWS AgentCore without legacy Bedrock Classic restrictions.
- **US-2**: As an API consumer, I want to invoke the AgentCore agent via `POST /agentcore/research` or via `BedrockAgentCoreRuntime` (`invoke_agent_runtime`) and receive validated `GroundedRAGResponse` outputs.
- **US-3**: As a DevOps operator, I want standard AgentCore ping/health endpoints (`/ping` returning `HEALTHY`) to monitor the running runtime.

### Acceptance Criteria
```gherkin
Scenario: BedrockAgentCoreApp handles agent invocation
  Given an invocation payload {"prompt": "What is multi-head attention?"}
  When the BedrockAgentCoreApp entrypoint function is invoked
  Then it executes the LangGraph workflow with AWS Bedrock foundation models
  And returns a JSON payload with {"result": GroundedRAGResponse, "session_id": string}

Scenario: AgentCore Runtime Client invokes remote AgentCore endpoint
  Given an agent runtime ARN "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/rag-agent"
  When BedrockAgentCoreRuntime.invoke_runtime is called with a prompt
  Then it calls boto3 invoke_agent_runtime with application/json payload
  And streams/reads the response body and parses GroundedRAGResponse

Scenario: DeepResearchAgent routes in agentcore mode
  Given DeepResearchAgent initialized with mode="agentcore"
  When run_research is called
  Then it dispatches to the Bedrock AgentCore runtime client or app entrypoint
```

---

## 3. Architecture & Data Contracts

### 1. Server Architecture (`src/aws/agentcore_app.py`)
```python
from bedrock_agentcore import BedrockAgentCoreApp, PingStatus

app = BedrockAgentCoreApp()

@app.entrypoint
def agentcore_entrypoint(payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # Extract prompt / query
    # Execute LangGraph workflow or direct research
    # Return structured dict
    ...

@app.ping
def ping_handler() -> PingStatus:
    return PingStatus.HEALTHY
```

### 2. Client Architecture (`src/aws/bedrock_agentcore_runtime.py`)
```python
class BedrockAgentCoreRuntime:
    def __init__(self, runtime_arn: Optional[str] = None, region_name: str = "us-east-1"):
        self.client = boto3.client("bedrock-agentcore", region_name=region_name)
        self.runtime_arn = runtime_arn
    
    def invoke_runtime(self, prompt: str, session_id: Optional[str] = None) -> GroundedRAGResponse:
        ...
```

---

## 4. Test Matrix

| Test ID | Target Component | Description | Expected Output |
| :--- | :--- | :--- | :--- |
| `test_agentcore_app_entrypoint_success` | `src/aws/agentcore_app.py` | Valid prompt through entrypoint | `{"result": GroundedRAGResponse, ...}` |
| `test_agentcore_app_entrypoint_empty` | `src/aws/agentcore_app.py` | Empty prompt handling | Structured error / fallback |
| `test_agentcore_app_ping` | `src/aws/agentcore_app.py` | Health ping handler | `PingStatus.HEALTHY` |
| `test_agentcore_runtime_client_invoke` | `src/aws/bedrock_agentcore_runtime.py` | Mocked Boto3 `invoke_agent_runtime` | Parsed `GroundedRAGResponse` |
| `test_deep_research_agent_agentcore_mode` | `src/main_agent.py` | Unified agent with `mode="agentcore"` | Executes AgentCore path |
