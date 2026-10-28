"""AWS Bedrock Agent Core Runtime Invocation Client.

Handles interaction with AWS Bedrock Agent Runtime (`invoke_agent`),
streaming response processing, trace extraction, and transformation
into the application's GroundedRAGResponse schema.
"""

import json
import logging
import os
import uuid
from typing import Any

from src.schemas import Citation, GroundedRAGResponse

logger = logging.getLogger(__name__)


class BedrockAgentRunner:
    """
    Client for invoking an AWS Bedrock Agent deployed with Bedrock Agent Core.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_alias_id: str | None = None,
        region_name: str | None = None,
        *,
        boto3_client: Any | None = None,
    ):
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        self.agent_id = agent_id or os.getenv("BEDROCK_AGENT_ID")
        self.agent_alias_id = agent_alias_id or os.getenv(
            "BEDROCK_AGENT_ALIAS_ID", "TSTALIASID"
        )

        if boto3_client is not None:
            self.client = boto3_client
        else:
            try:
                import boto3

                self.client = boto3.client(
                    "bedrock-agent-runtime",
                    region_name=self.region_name,
                )
            except Exception as e:
                logger.warning(
                    f"Could not initialize bedrock-agent-runtime client: {e}. "
                    "Ensure boto3 is installed and AWS credentials are configured."
                )
                self.client = None

    def invoke_agent(
        self,
        query: str,
        session_id: str | None = None,
        *,
        enable_trace: bool = True,
        end_session: bool = False,
    ) -> GroundedRAGResponse:
        """
        Invokes the Bedrock Agent with a user query and processes the streaming event output.
        Returns a validated GroundedRAGResponse.
        """
        if not self.agent_id:
            raise ValueError(
                "BEDROCK_AGENT_ID is not configured. Set the environment variable or pass agent_id."
            )
        if self.client is None:
            raise RuntimeError(
                "Bedrock Agent Runtime client is not initialized. Check AWS credentials."
            )

        active_session_id = session_id or str(uuid.uuid4())

        try:
            response = self.client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=active_session_id,
                inputText=query,
                enableTrace=enable_trace,
                endSession=end_session,
            )
        except Exception as e:
            logger.error(f"Error calling Bedrock Agent invoke_agent: {e}")
            return GroundedRAGResponse(
                answer=f"Bedrock Agent execution error: {e}",
                citations=[],
                confidence_score=0.0,
            )

        # Process the streaming response
        raw_text, traces, citations = self._process_event_stream(response.get("completion", []))

        # Attempt to parse as structured JSON first
        structured_response = self._parse_structured_or_fallback(raw_text, citations)
        return structured_response

    def _process_event_stream(
        self, event_stream: Any
    ) -> tuple[str, list[dict[str, Any]], list[Citation]]:
        """Processes chunks, citations, and traces from Bedrock Agent's streaming completion."""
        text_chunks: list[str] = []
        traces: list[dict[str, Any]] = []
        citations: list[Citation] = []

        for event in event_stream:
            # 1. Text chunk
            if "chunk" in event:
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    chunk_text = chunk_data["bytes"].decode("utf-8")
                    text_chunks.append(chunk_text)

                # Attributions (Citations generated natively by Bedrock Agent)
                if "attribution" in chunk_data:
                    for cit in chunk_data["attribution"].get("citations", []):
                        for ref in cit.get("retrievedReferences", []):
                            location = ref.get("location", {})
                            content = ref.get("content", {}).get("text", "")
                            s3_uri = location.get("s3Location", {}).get("uri", "")
                            citations.append(
                                Citation(
                                    chunk_id=ref.get("metadata", {}).get("chunk_id", "bedrock_ref"),
                                    source_doc=s3_uri or location.get("type", "bedrock_kb"),
                                    relevant_snippet=content[:200] if content else "Retrieved evidence",
                                )
                            )

            # 2. Execution trace (step-by-step reasoning / action group calls)
            elif "trace" in event:
                traces.append(event["trace"])

        full_text = "".join(text_chunks).strip()
        return full_text, traces, citations

    def _parse_structured_or_fallback(
        self, text: str, native_citations: list[Citation]
    ) -> GroundedRAGResponse:
        """Parses output into GroundedRAGResponse schema."""
        if not text:
            return GroundedRAGResponse(
                answer="No response generated by Bedrock Agent.",
                citations=[],
                confidence_score=0.0,
            )

        # Check for JSON block
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text.strip()

            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "answer" in parsed:
                return GroundedRAGResponse(**parsed)
        except Exception:
            pass

        # If plain text, wrap with any extracted native citations
        return GroundedRAGResponse(
            answer=text,
            citations=native_citations,
            confidence_score=0.85 if native_citations else 0.5,
        )

