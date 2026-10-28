"""AWS Lambda Action Group Handler for AWS Bedrock Agent Core.

This Lambda is invoked by Amazon Bedrock Agent to perform GraphRAG operations:
- hybrid_search: Dense pgvector + Sparse BM25 retrieval
- neo4j_traverse: Knowledge Graph ontology multi-hop traversal
- retrieve_and_rerank: Full hybrid retrieval + KG augmentation + Cross-Encoder reranking
- sanitize_input: Guardrails PII redaction and input sanitation
"""

import json
import logging
import os
from typing import Any

from src.guardrails import InputGuardrail
from src.state import RAGState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Lazy singletons for retriever and neo4j client to optimize Lambda warm starts
_retriever = None
_neo4j_client = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval import ProductionGraphRAGRetriever
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag",
        )
        _retriever = ProductionGraphRAGRetriever(db_url=db_url)
    return _retriever


def get_neo4j_client():
    global _neo4j_client
    if _neo4j_client is None:
        from src.neo4j_client import Neo4jGraphClient
        _neo4j_client = Neo4jGraphClient()
    return _neo4j_client


def _extract_parameters(event: dict[str, Any]) -> dict[str, Any]:
    """Extracts parameters from either function schema, API schema, or requestBody."""
    params: dict[str, Any] = {}

    # 1. From Bedrock function parameters list
    for param in event.get("parameters", []):
        name = param.get("name")
        val = param.get("value")
        if name:
            params[name] = val

    # 2. From Bedrock requestBody (OpenAPI schema style)
    request_body = event.get("requestBody", {})
    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    body_props = json_content.get("properties", [])
    if isinstance(body_props, list):
        for prop in body_props:
            name = prop.get("name")
            val = prop.get("value")
            if name:
                params[name] = val
    elif isinstance(json_content, dict) and "body" in json_content:
        try:
            parsed = json.loads(json_content["body"])
            if isinstance(parsed, dict):
                params.update(parsed)
        except Exception:
            pass

    return params


def handle_hybrid_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Runs dense pgvector vector search + sparse keyword search."""
    retriever = get_retriever()
    safe_query = InputGuardrail.sanitize_and_redact(query)
    candidates = retriever._run_hybrid_search(safe_query, top_k=top_k)
    return candidates


def handle_neo4j_traverse(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Runs ontology-aware multi-hop traversal in Neo4j."""
    neo4j_client = get_neo4j_client()
    safe_query = InputGuardrail.sanitize_and_redact(query)
    kg_chunks = neo4j_client.ontology_multihop_traverse(safe_query, limit=limit)
    return kg_chunks


def handle_retrieve_and_rerank(query: str, top_k: int = 3) -> dict[str, Any]:
    """Runs end-to-end GraphRAG retrieval pipeline with cross-encoder reranking."""
    retriever = get_retriever()
    safe_query = InputGuardrail.sanitize_and_redact(query)
    state = RAGState(query=safe_query)
    result_state = retriever.retrieve(state, top_k=top_k)
    return {
        "query": safe_query,
        "retrieved_candidates": result_state.retrieved_candidates,
        "count": len(result_state.retrieved_candidates),
    }


def handle_sanitize_input(query: str) -> dict[str, str]:
    """Sanitizes query and redacts PII."""
    safe_query = InputGuardrail.sanitize_and_redact(query)
    return {"safe_query": safe_query}


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda entry point for Bedrock Agent Action Group.
    Dispatches to appropriate retrieval/graph operations and wraps the result in
    the Bedrock Agent response specification.
    """
    logger.info(f"Received Bedrock Agent Action Group event: {json.dumps(event)}")

    action_group = event.get("actionGroup", "GraphRAGActionGroup")
    api_path = event.get("apiPath") or ""
    function_name = event.get("function") or ""
    http_method = event.get("httpMethod", "POST")

    # Match operation either by function name or api path
    operation = function_name or api_path.strip("/")
    params = _extract_parameters(event)
    query = params.get("query", "")
    top_k = int(params.get("top_k", 3))

    status_code = 200
    try:
        if operation in ("retrieve_and_rerank", "retrieve", "/retrieve_and_rerank"):
            result_data = handle_retrieve_and_rerank(query=query, top_k=top_k)
        elif operation in ("hybrid_search", "/hybrid_search"):
            result_data = handle_hybrid_search(query=query, top_k=top_k)
        elif operation in ("neo4j_traverse", "graph_traversal", "/neo4j_traverse"):
            result_data = handle_neo4j_traverse(query=query, limit=top_k)
        elif operation in ("sanitize_input", "/sanitize_input"):
            result_data = handle_sanitize_input(query=query)
        else:
            # Default fallback: full retrieve and rerank
            result_data = handle_retrieve_and_rerank(query=query, top_k=top_k)

        response_body = {"status": "success", "data": result_data}
    except Exception as e:
        logger.error(f"Error handling action group operation '{operation}': {e}", exc_info=True)
        status_code = 500
        response_body = {"status": "error", "error": str(e)}

    # Construct Bedrock Agent Action Group response
    response_payload = {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "apiPath": api_path if api_path else None,
            "function": function_name if function_name else None,
            "httpMethod": http_method,
            "httpStatusCode": status_code,
            "responseBody": {
                "application/json": {
                    "body": json.dumps(response_body)
                }
            },
        },
    }

    # Clean None values in response
    response_payload["response"] = {
        k: v for k, v in response_payload["response"].items() if v is not None
    }

    return response_payload

