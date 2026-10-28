# Feature Specification: Local LangGraph Multi-Step ReAct Agent Workflow

> **Status**: Implemented  
> **Author**: DeepMind Advanced Agentic Coding  
> **Date**: 2026-10-28  
> **Spec ID**: `specs/002-langgraph-agent-workflow`  
> **Related Files**: [`src/agent_workflow.py`](file:///Users/sabyasachi/rag-evaluation-engine/src/agent_workflow.py), [`src/main_agent.py`](file:///Users/sabyasachi/rag-evaluation-engine/src/main_agent.py), [`tests/test_agent_workflow.py`](file:///Users/sabyasachi/rag-evaluation-engine/tests/test_agent_workflow.py)

---

## 1. Problem Statement & Context
- **Background**: Previously, local agent execution was either a single-pass orchestrator run or hard-coded linear steps, while AWS Bedrock Agent Core operates as a dynamic multi-step ReAct agent in the cloud.
- **Objective**: Build a local autonomous multi-step ReAct agent using **LangGraph** (`StateGraph`) that executes locally with zero cloud costs, shares the exact same 4 tool interfaces as AWS Bedrock Agent Core Action Groups (`retrieve_and_rerank`, `hybrid_search`, `neo4j_traverse`, `sanitize_input`), implements reflection & query broadening, and includes circuit-breaker resilience.
- **Scope**: Local multi-step execution, dynamic tool dispatch, reflection loops, query broadening fallback, and unified routing via `DeepResearchAgent(mode="local")`.

---

## 2. User Stories & Acceptance Criteria

### User Stories
- **US-1**: As a researcher/developer, I want to run the multi-step research agent locally so I can test and evaluate multi-hop GraphRAG reasoning without incurring AWS Bedrock charges.
- **US-2**: As an engineer, I want the local agent tools to mirror the AWS Bedrock Action Group OpenAPI specification 1-to-1 so local logic transitions seamlessly to the cloud.
- **US-3**: As a system operator, I want the agent to automatically broaden search queries when primary retrieval yields zero context, and gracefully handle database timeouts with circuit breakers.

### Acceptance Criteria
```gherkin
Scenario: Successful local multi-step research
  Given a user question "How does multi-head attention work?"
  When DeepResearchAgent.run_research is invoked locally
  Then the LangGraph StateGraph executes: input_guardrail -> planner -> execute_tools -> reflection -> synthesizer
  And returns a validated GroundedRAGResponse with citations

Scenario: Query broadening on empty evidence
  Given a query where initial retrieval yields 0 chunks
  When the reflection node evaluates the state
  Then it flags is_sufficient = False
  And broadens the query to "Overview of <query>"
  And re-executes retrieval before synthesizing the answer

Scenario: Database failure circuit breaking
  Given a database timeout or connection error in retrieve_and_rerank
  When execute_tools runs
  Then it catches the exception and falls back to hybrid/sparse search
  And completes workflow without unhandled crash
```

---

## 3. Architecture & Data Contracts

### StateGraph Schema
```python
class AgentWorkflowState(TypedDict, total=False):
    raw_query: str
    safe_query: str
    research_query: str
    iteration: int
    max_iterations: int
    selected_tool: str
    retrieved_chunks: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    reflection_notes: str
    is_sufficient: bool
    final_response: GroundedRAGResponse
```

### Node Topology
```
[input_guardrail] ──> [planner] ──> [execute_tools] ──> [reflection]
                                                           │
                                        ┌──────────────────┴──────────────────┐
                                        ▼ (is_sufficient or max_iter)          ▼ (insufficient & < max_iter)
                                   [synthesizer] ──> END                  [planner] (loop)
```

---

## 4. Test Matrix & Verification

| Test Case | Description | Tested Component | Expected Result | Status |
|---|---|---|---|---|
| `TC-01` | Happy path execution | `LocalGraphRAGWorkflow.run` | Returns `GroundedRAGResponse` with citations | PASSED |
| `TC-02` | PII Redaction in workflow | `node_input_guardrail` | Replaces emails with `[REDACTED_EMAIL]` before DB search | PASSED |
| `TC-03` | Query broadening on empty context | `node_reflection` | Triggers 2nd retrieval with "Overview of ..." | PASSED |
| `TC-04` | Circuit breaker on retriever failure | `tool_retrieve_and_rerank` | Gracefully falls back to hybrid search | PASSED |

---

## 5. Deployment & Integration Parity
- **Local Mode**: `AGENT_MODE=local` or `DeepResearchAgent(mode="local")` $\rightarrow$ triggers `LocalGraphRAGWorkflow`.
- **AWS Bedrock Mode**: `AGENT_MODE=bedrock_agent` $\rightarrow$ triggers `BedrockAgentRunner` invoking AWS Bedrock Agent Core in the cloud.
- **Unified Schema**: Both emit [`GroundedRAGResponse`](file:///Users/sabyasachi/rag-evaluation-engine/src/schemas.py#L9-L12).

