#!/usr/bin/env python3
"""Automated Deployment Script for AWS Bedrock Agent Core + AWS Bedrock.

This script provisions or updates:
1. Bedrock Agent IAM Execution Role
2. Lambda Action Group Execution Role
3. Lambda Function for the GraphRAG Action Group
4. Bedrock Agent configured with foundation model (e.g. Claude 3.5 Sonnet)
5. Action Group attaching the Lambda function and OpenAPI 3.0 schema
6. Agent Preparation & Production Alias
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy_bedrock_agent")


def load_openapi_schema(schema_path: Path) -> str:
    """Reads the OpenAPI schema as a JSON string."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read()


def build_lambda_zip(repo_root: Path) -> bytes:
    """Builds an in-memory zip archive containing the application code for the Lambda function."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Include src directory
        src_dir = repo_root / "src"
        for file in src_dir.rglob("*.py"):
            arcname = file.relative_to(repo_root)
            zip_file.write(file, arcname=str(arcname))
        for file in src_dir.rglob("*.json"):
            arcname = file.relative_to(repo_root)
            zip_file.write(file, arcname=str(arcname))

    zip_buffer.seek(0)
    return zip_buffer.read()


def setup_bedrock_agent_role(iam_client: Any, account_id: str, role_name: str) -> str:
    """Creates or updates the IAM execution role for Bedrock Agent Core."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        logger.info(f"Using existing Bedrock Agent IAM role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"Creating Bedrock Agent IAM role: {role_name}...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for Bedrock Agent Core GraphRAG Agent",
        )
        role_arn = response["Role"]["Arn"]
        logger.info(f"Created Bedrock Agent IAM role: {role_arn}")

    # Attach inline policy to invoke Bedrock foundation models
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetFoundationModel",
                ],
                "Resource": "*",
            }
        ],
    }
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="BedrockFoundationModelAccess",
        PolicyDocument=json.dumps(policy_doc),
    )
    return role_arn


def setup_lambda_role(iam_client: Any, role_name: str) -> str:
    """Creates or retrieves the IAM role for the Action Group Lambda function."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        logger.info(f"Using existing Lambda IAM role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"Creating Lambda IAM role: {role_name}...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for GraphRAG Bedrock Action Group Lambda",
        )
        role_arn = response["Role"]["Arn"]

    # Attach AWSLambdaBasicExecutionRole
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    return role_arn


def deploy_lambda_function(
    lambda_client: Any,
    function_name: str,
    role_arn: str,
    zip_bytes: bytes,
    environment_vars: Dict[str, str],
) -> str:
    """Creates or updates the Action Group Lambda function."""
    try:
        response = lambda_client.get_function(FunctionName=function_name)
        func_arn = response["Configuration"]["FunctionArn"]
        logger.info(f"Updating code for existing Lambda function: {func_arn}")
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_bytes,
        )
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment={"Variables": environment_vars},
            Timeout=60,
            MemorySize=512,
        )
        return func_arn
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.info(f"Creating Lambda function: {function_name}...")
        # Brief pause for IAM role propagation
        time.sleep(5)
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime="python3.11",
            Role=role_arn,
            Handler="src.aws.lambda_action_group.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Description="GraphRAG Action Group tool handler for AWS Bedrock Agent",
            Timeout=60,
            MemorySize=512,
            Environment={"Variables": environment_vars},
        )
        return response["FunctionArn"]


def grant_bedrock_permission_to_lambda(
    lambda_client: Any, function_name: str, agent_arn_pattern: str
) -> None:
    """Grants bedrock.amazonaws.com permission to invoke the Lambda action group."""
    statement_id = "BedrockAgentInvocationPermission"
    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="bedrock.amazonaws.com",
            SourceArn=agent_arn_pattern,
        )
        logger.info("Granted Bedrock permission to invoke Lambda.")
    except lambda_client.exceptions.ResourceConflictException:
        logger.info("Bedrock invoke permission already present on Lambda function.")


def deploy_or_update_bedrock_agent(
    bedrock_agent_client: Any,
    agent_name: str,
    agent_role_arn: str,
    model_id: str,
    instruction: str,
) -> str:
    """Creates or updates the AWS Bedrock Agent."""
    # List agents to find existing by name
    paginator = bedrock_agent_client.get_paginator("list_agents")
    agent_id = None
    for page in paginator.paginate():
        for summary in page.get("agentSummaries", []):
            if summary.get("agentName") == agent_name:
                agent_id = summary.get("agentId")
                break
        if agent_id:
            break

    if agent_id:
        logger.info(f"Updating existing Bedrock Agent (ID: {agent_id})...")
        response = bedrock_agent_client.update_agent(
            agentId=agent_id,
            agentName=agent_name,
            agentResourceRoleArn=agent_role_arn,
            foundationModel=model_id,
            instruction=instruction,
        )
    else:
        logger.info(f"Creating new Bedrock Agent: {agent_name}...")
        response = bedrock_agent_client.create_agent(
            agentName=agent_name,
            agentResourceRoleArn=agent_role_arn,
            foundationModel=model_id,
            instruction=instruction,
            description="Production GraphRAG Deep Research Agent powered by AWS Bedrock Agent Core",
        )
        agent_id = response["agent"]["agentId"]

    logger.info(f"Bedrock Agent ready with ID: {agent_id}")
    return agent_id


def deploy_or_update_action_group(
    bedrock_agent_client: Any,
    agent_id: str,
    action_group_name: str,
    lambda_arn: str,
    openapi_schema_json: str,
) -> str:
    """Creates or updates the Action Group on the Bedrock Agent."""
    action_group_id = None
    try:
        response = bedrock_agent_client.list_agent_action_groups(
            agentId=agent_id,
            agentVersion="DRAFT",
        )
        for ag in response.get("actionGroupSummaries", []):
            if ag.get("actionGroupName") == action_group_name:
                action_group_id = ag.get("actionGroupId")
                break
    except Exception as e:
        logger.warning(f"Could not list action groups: {e}")

    api_schema = {"payload": openapi_schema_json}
    action_group_executor = {"lambda": lambda_arn}

    if action_group_id:
        logger.info(f"Updating Action Group: {action_group_name} (ID: {action_group_id})...")
        response = bedrock_agent_client.update_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupId=action_group_id,
            actionGroupName=action_group_name,
            actionGroupExecutor=action_group_executor,
            apiSchema=api_schema,
            actionGroupState="ENABLED",
        )
    else:
        logger.info(f"Creating Action Group: {action_group_name}...")
        response = bedrock_agent_client.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupName=action_group_name,
            actionGroupExecutor=action_group_executor,
            apiSchema=api_schema,
            actionGroupState="ENABLED",
            description="GraphRAG hybrid retrieval and ontology multi-hop traversal tools",
        )
        action_group_id = response["agentActionGroup"]["actionGroupId"]

    logger.info(f"Action Group configured with ID: {action_group_id}")
    return action_group_id


def prepare_agent_and_create_alias(
    bedrock_agent_client: Any,
    agent_id: str,
    alias_name: str = "production",
) -> str:
    """Prepares the agent and creates or updates a production alias."""
    logger.info("Preparing Bedrock Agent (compiling DRAFT version)...")
    bedrock_agent_client.prepare_agent(agentId=agent_id)

    # Wait for PREPARED status
    for _ in range(12):
        time.sleep(3)
        desc = bedrock_agent_client.get_agent(agentId=agent_id)
        status = desc["agent"]["agentStatus"]
        logger.info(f"Agent status: {status}")
        if status in ("PREPARED", "NOT_PREPARED"):
            break

    # Look for alias
    alias_id = None
    try:
        aliases = bedrock_agent_client.list_agent_aliases(agentId=agent_id)
        for a in aliases.get("agentAliasSummaries", []):
            if a.get("agentAliasName") == alias_name:
                alias_id = a.get("agentAliasId")
                break
    except Exception as e:
        logger.warning(f"Error checking aliases: {e}")

    if alias_id:
        logger.info(f"Using existing Agent Alias '{alias_name}' (ID: {alias_id})")
    else:
        logger.info(f"Creating new Agent Alias: {alias_name}...")
        res = bedrock_agent_client.create_agent_alias(
            agentId=agent_id,
            agentAliasName=alias_name,
            description="Production alias for GraphRAG Agent",
        )
        alias_id = res["agentAlias"]["agentAliasId"]

    return alias_id


def main():
    parser = argparse.ArgumentParser(
        description="Deploy GraphRAG Deep Research Agent to AWS Bedrock Agent Core"
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        help="AWS Bedrock Foundation Model ID",
    )
    parser.add_argument(
        "--agent-name",
        default="rag-graph-research-agent",
        help="Name of the Bedrock Agent",
    )
    parser.add_argument(
        "--alias-name",
        default="production",
        help="Alias name (default: production)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without making AWS API calls",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    openapi_path = repo_root / "src" / "aws" / "action_group_openapi.json"

    logger.info("=== Deploying GraphRAG Agent to AWS Bedrock Agent Core ===")
    logger.info(f"AWS Region: {args.region}")
    logger.info(f"Bedrock Model: {args.model_id}")
    logger.info(f"Agent Name: {args.agent_name}")

    if args.dry_run:
        logger.info("[Dry Run] Validation complete. No AWS resources modified.")
        return

    import boto3

    session = boto3.Session(region_name=args.region)
    sts_client = session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    logger.info(f"Authenticated as AWS Account: {account_id}")

    iam_client = session.client("iam")
    lambda_client = session.client("lambda")
    bedrock_agent_client = session.client("bedrock-agent")

    # 1. IAM Roles
    bedrock_role_arn = setup_bedrock_agent_role(
        iam_client=iam_client,
        account_id=account_id,
        role_name="BedrockAgentGraphRAGRole",
    )
    lambda_role_arn = setup_lambda_role(
        iam_client=iam_client,
        role_name="GraphRAGActionGroupLambdaRole",
    )

    # 2. Package & Deploy Action Group Lambda
    logger.info("Packaging Lambda zip bundle...")
    zip_bytes = build_lambda_zip(repo_root)
    env_vars = {
        "DATABASE_URL": os.getenv("DATABASE_URL", ""),
        "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", ""),
    }
    lambda_arn = deploy_lambda_function(
        lambda_client=lambda_client,
        function_name="rag-graph-action-group",
        role_arn=lambda_role_arn,
        zip_bytes=zip_bytes,
        environment_vars=env_vars,
    )
    logger.info(f"Lambda Action Group deployed at: {lambda_arn}")

    # 3. Agent Instruction
    agent_instruction = (
        "You are an expert GraphRAG Deep Research Agent. "
        "Your task is to answer user queries with high technical accuracy by retrieving evidence from the knowledge graph and vector database. "
        "Always call the 'GraphRAGActionGroup' tools (specifically 'retrieve_and_rerank') to fetch relevant context before answering. "
        "Base your answers STRICTLY on the retrieved context chunks and ontology relationships. "
        "You MUST output your response in valid JSON conforming to this schema:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "citations": [{"chunk_id": "string", "source_doc": "string", "relevant_snippet": "string"}],\n'
        '  "confidence_score": float\n'
        "}"
    )

    # 4. Deploy Bedrock Agent
    agent_id = deploy_or_update_bedrock_agent(
        bedrock_agent_client=bedrock_agent_client,
        agent_name=args.agent_name,
        agent_role_arn=bedrock_role_arn,
        model_id=args.model_id,
        instruction=agent_instruction,
    )

    # 5. Grant invoke permission
    grant_bedrock_permission_to_lambda(
        lambda_client=lambda_client,
        function_name="rag-graph-action-group",
        agent_arn_pattern=f"arn:aws:bedrock:{args.region}:{account_id}:agent/{agent_id}",
    )

    # 6. Deploy Action Group
    openapi_schema_json = load_openapi_schema(openapi_path)
    deploy_or_update_action_group(
        bedrock_agent_client=bedrock_agent_client,
        agent_id=agent_id,
        action_group_name="GraphRAGActionGroup",
        lambda_arn=lambda_arn,
        openapi_schema_json=openapi_schema_json,
    )

    # 7. Prepare Agent & Setup Alias
    alias_id = prepare_agent_and_create_alias(
        bedrock_agent_client=bedrock_agent_client,
        agent_id=agent_id,
        alias_name=args.alias_name,
    )

    logger.info("\n================ DEPLOYMENT SUCCESSFUL ================")
    logger.info(f"BEDROCK_AGENT_ID={agent_id}")
    logger.info(f"BEDROCK_AGENT_ALIAS_ID={alias_id}")
    logger.info(f"BEDROCK_MODEL_ID={args.model_id}")
    logger.info(f"AWS_REGION={args.region}")
    logger.info("=======================================================\n")
    logger.info("Add these values to your .env or CI/CD secrets.")


if __name__ == "__main__":
    main()

