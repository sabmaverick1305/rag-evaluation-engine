import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from src.schemas import GroundedRAGResponse

logger = logging.getLogger(__name__)


class BedrockGenerator:
    """
    Production Generator powered by AWS Bedrock foundation models (e.g. Anthropic Claude, Amazon Nova).
    Replaces local HuggingFace/Qwen generation with scalable cloud-managed inference.
    """

    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        boto3_client: Any | None = None,
    ):
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        self.model_id = (
            model_id
            or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        )
        self.temperature = float(os.getenv("BEDROCK_TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", str(max_tokens)))

        if boto3_client is not None:
            self.client = boto3_client
        else:
            try:
                import boto3

                self.client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region_name,
                )
            except Exception as e:
                logger.warning(
                    f"Could not initialize bedrock-runtime boto3 client: {e}. "
                    "In local or mock mode, supply boto3_client directly."
                )
                self.client = None

    def _call_converse(self, system_prompt: str, user_prompt: str) -> str:
        """Invokes Bedrock using the unified Converse API."""
        if self.client is None:
            raise RuntimeError(
                "AWS Bedrock client is not initialized. Check AWS credentials and region."
            )

        messages = [
            {
                "role": "user",
                "content": [{"text": user_prompt}],
            }
        ]
        system = [{"text": system_prompt}]

        response = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=system,
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        text_parts = [
            block.get("text", "") for block in content_blocks if "text" in block
        ]
        return "".join(text_parts).strip()

    def _call_invoke_model_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback invocation using invoke_model for Claude or other model families."""
        if self.client is None:
            raise RuntimeError("AWS Bedrock client is not initialized.")

        # Claude Messages format
        if "anthropic.claude" in self.model_id:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            body = json.loads(response["body"].read().decode("utf-8"))
            content = body.get("content", [])
            return "".join([c.get("text", "") for c in content if c.get("type") == "text"]).strip()

        # Amazon Nova / Titan format
        elif "amazon.nova" in self.model_id:
            payload = {
                "system": [{"text": system_prompt}],
                "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
                "inferenceConfig": {"max_new_tokens": self.max_tokens, "temperature": self.temperature},
            }
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            body = json.loads(response["body"].read().decode("utf-8"))
            return (
                body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip()
            )
        else:
            # Generic fallback
            payload = {
                "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
                "max_tokens_to_sample": self.max_tokens,
                "temperature": self.temperature,
            }
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            body = json.loads(response["body"].read().decode("utf-8"))
            return body.get("completion", "").strip()

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> GroundedRAGResponse:
        """
        Generates a grounded technical response strictly adhering to GroundedRAGResponse schema
        using AWS Bedrock foundation models.
        """
        if diagnostics is not None:
            diagnostics.update(
                raw_output=None,
                schema_valid=False,
                generation_skipped=False,
                validation_failure=None,
            )

        if not retrieved_chunks or retrieved_chunks[0].get("score", 1.0) < -2.0:
            if diagnostics is not None:
                diagnostics["generation_skipped"] = True
            return GroundedRAGResponse(
                answer="I cannot answer this based on the provided documents as no sufficiently relevant context was found.",
                citations=[],
                confidence_score=0.0,
            )

        context_str = "\n\n".join([
            f"--- CHUNK ID: {chunk['id']} (Source: {chunk.get('source_doc', 'doc')}) ---\n{chunk.get('text', '')}"
            for chunk in retrieved_chunks
        ])

        system_prompt = (
            "You are a precise technical AI assistant. Answer the user's question using ONLY the provided context. "
            "You MUST output valid, parseable JSON matching this exact schema and nothing else:\n"
            "{\n"
            "  \"answer\": \"string\",\n"
            "  \"citations\": [\n"
            "    {\"chunk_id\": \"string\", \"source_doc\": \"string\", \"relevant_snippet\": \"string\"}\n"
            "  ],\n"
            "  \"confidence_score\": float\n"
            "}"
        )

        user_prompt = f"Context:\n{context_str}\n\nUser Question: {query}"

        # Invoke Bedrock (prefer Converse API, fallback to invoke_model)
        try:
            response_raw = self._call_converse(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"Bedrock Converse API call failed ({e}); attempting invoke_model fallback...")
            try:
                response_raw = self._call_invoke_model_fallback(system_prompt, user_prompt)
            except Exception as e2:
                logger.error(f"Both Bedrock Converse and invoke_model failed: {e2}")
                if diagnostics is not None:
                    diagnostics["validation_failure"] = {
                        "type": type(e2).__name__,
                        "message": str(e2),
                    }
                return GroundedRAGResponse(
                    answer=f"Bedrock generation failed: {e2}",
                    citations=[],
                    confidence_score=0.0,
                )

        if diagnostics is not None:
            diagnostics["raw_output"] = response_raw

        # Extract & parse JSON block
        try:
            if "```json" in response_raw:
                json_str = response_raw.split("```json")[1].split("```")[0].strip()
            elif "```" in response_raw:
                json_str = response_raw.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_raw.strip()

            data = json.loads(json_str)
            response = GroundedRAGResponse(**data)
            if diagnostics is not None:
                diagnostics["schema_valid"] = True
            return response
        except (json.JSONDecodeError, ValidationError) as err:
            if diagnostics is not None:
                diagnostics["validation_failure"] = {
                    "type": type(err).__name__,
                    "message": str(err),
                }
            return GroundedRAGResponse(
                answer=response_raw.strip(),
                citations=[],
                confidence_score=0.0,
            )

