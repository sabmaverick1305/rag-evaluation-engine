"""Amazon Bedrock AgentCore Runtime Client.

Invokes the deployed agent runtime hosted on Amazon Bedrock AgentCore
using boto3 invoke_agent_runtime.
"""

import json
import logging
import os
from typing import Any, Optional
import uuid

import boto3
from src.schemas import GroundedRAGResponse

logger = logging.getLogger(__name__)


class BedrockAgentCoreRuntime:
    """Client for invoking autonomous agents hosted on Amazon Bedrock AgentCore Runtime."""

    def __init__(
        self,
        runtime_arn: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.runtime_arn = runtime_arn or os.getenv("BEDROCK_AGENTCORE_RUNTIME_ARN", "")
        self.client = client or boto3.client("bedrock-agentcore", region_name=self.region_name)

    def invoke_runtime(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> GroundedRAGResponse:
        """Invokes the remote AgentCore runtime with the user prompt.

        Args:
            prompt: User question or research query.
            session_id: Optional session identifier for cross-turn state.

        Returns:
            GroundedRAGResponse: Verified answer with citations and confidence score.
        """
        if not self.runtime_arn:
            raise ValueError(
                "BEDROCK_AGENTCORE_RUNTIME_ARN is required to invoke the Bedrock AgentCore Runtime. "
                "Specify runtime_arn or set the BEDROCK_AGENTCORE_RUNTIME_ARN environment variable."
            )

        sid = session_id or str(uuid.uuid4())
        payload_data = {
            "prompt": prompt,
            "session_id": sid,
        }
        payload_bytes = json.dumps(payload_data).encode("utf-8")

        logger.info(f"Invoking Bedrock AgentCore Runtime: {self.runtime_arn} (session: {sid})")
        try:
            response = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.runtime_arn,
                runtimeSessionId=sid,
                contentType="application/json",
                accept="application/json",
                payload=payload_bytes,
            )

            status_code = response.get("statusCode", 200)
            if status_code != 200:
                logger.warning(f"Bedrock AgentCore returned non-200 status: {status_code}")

            raw_body = response["response"].read().decode("utf-8")
            data = json.loads(raw_body)

            # Unpack result if wrapped in AgentCore envelope
            res = data.get("result", data)
            return GroundedRAGResponse.model_validate(res)

        except Exception as exc:
            logger.error(f"Failed to invoke Bedrock AgentCore Runtime: {exc}", exc_info=True)
            raise
