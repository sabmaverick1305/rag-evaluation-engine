#!/usr/bin/env python3
"""Automated Deployment Script for AWS App Runner.

Provisions:
1. App Runner ECR Access IAM Role (build.apprunner.amazonaws.com)
2. App Runner Instance Execution Role (tasks.apprunner.amazonaws.com) with Bedrock access
3. App Runner Service hosting the production GraphRAG Deep Research Agent container from ECR
"""

import json
import logging
import sys
import time
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy_apprunner")


def setup_ecr_access_role(iam_client: Any) -> str:
    """Creates or verifies the IAM role used by App Runner to pull images from private ECR."""
    role_name = "AppRunnerECRAccessRole"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "build.apprunner.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        logger.info(f"Using existing ECR access role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"Creating ECR access role: {role_name}...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role allowing App Runner to pull images from private ECR",
        )
        role_arn = response["Role"]["Arn"]

    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess",
    )
    return role_arn


def setup_instance_role(iam_client: Any) -> str:
    """Creates or verifies the runtime IAM role assumed by the container in App Runner."""
    role_name = "AppRunnerInstanceRole"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "tasks.apprunner.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        logger.info(f"Using existing App Runner instance role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"Creating App Runner instance role: {role_name}...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Runtime role for App Runner container calling Bedrock and Lambda",
        )
        role_arn = response["Role"]["Arn"]

    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockConverseAccess",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetInferenceProfile",
                    "bedrock:ListInferenceProfiles",
                ],
                "Resource": "*",
            },
            {
                "Sid": "LambdaToolsAccess",
                "Effect": "Allow",
                "Action": [
                    "lambda:InvokeFunction",
                ],
                "Resource": "arn:aws:lambda:*:*:function:rag-graph-action-group",
            },
        ],
    }

    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="AppRunnerBedrockAndLambdaAccess",
        PolicyDocument=json.dumps(policy_doc),
    )
    return role_arn


def deploy_apprunner_service(
    apprunner_client: Any,
    service_name: str,
    image_uri: str,
    access_role_arn: str,
    instance_role_arn: str,
    region: str = "us-east-1",
) -> Dict[str, Any]:
    """Provisions or updates the App Runner service."""
    # Check if service already exists
    service_arn = None
    next_token = None
    while True:
        kwargs = {}
        if next_token:
            kwargs["NextToken"] = next_token
        res = apprunner_client.list_services(**kwargs)
        for summary in res.get("ServiceSummaryList", []):
            if summary.get("ServiceName") == service_name:
                service_arn = summary.get("ServiceArn")
                break
        if service_arn or "NextToken" not in res:
            break
        next_token = res["NextToken"]

    env_vars = {
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "LLM_PROVIDER": "bedrock",
        "AGENT_MODE": "agentcore",
        "PYTHONUNBUFFERED": "1",
    }

    source_config = {
        "ImageRepository": {
            "ImageIdentifier": image_uri,
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8000",
                "RuntimeEnvironmentVariables": env_vars,
            },
        },
        "AutoDeploymentsEnabled": False,
        "AuthenticationConfiguration": {
            "AccessRoleArn": access_role_arn,
        },
    }

    instance_config = {
        "Cpu": "2048",
        "Memory": "4096",
        "InstanceRoleArn": instance_role_arn,
    }

    health_check_config = {
        "Protocol": "HTTP",
        "Path": "/health",
        "Interval": 10,
        "Timeout": 5,
        "HealthyThreshold": 1,
        "UnhealthyThreshold": 5,
    }

    if service_arn:
        logger.info(f"Updating existing App Runner service: {service_arn}")
        response = apprunner_client.update_service(
            ServiceArn=service_arn,
            SourceConfiguration=source_config,
            InstanceConfiguration=instance_config,
            HealthCheckConfiguration=health_check_config,
        )
        service = response["Service"]
    else:
        logger.info(f"Creating new App Runner service: {service_name}...")
        for attempt in range(8):
            try:
                response = apprunner_client.create_service(
                    ServiceName=service_name,
                    SourceConfiguration=source_config,
                    InstanceConfiguration=instance_config,
                    HealthCheckConfiguration=health_check_config,
                )
                service = response["Service"]
                break
            except ClientError as e:
                if attempt < 7 and ("role" in str(e).lower() or "assume" in str(e).lower()):
                    logger.info("Waiting for IAM roles to propagate to App Runner (5s)...")
                    time.sleep(5)
                else:
                    raise

    return service


def wait_for_service(apprunner_client: Any, service_arn: str, timeout_seconds: int = 600) -> Dict[str, Any]:
    """Polls App Runner service until it reaches RUNNING or fails."""
    start_time = time.time()
    logger.info("Waiting for App Runner service deployment to complete...")
    while time.time() - start_time < timeout_seconds:
        desc = apprunner_client.describe_service(ServiceArn=service_arn)["Service"]
        status = desc.get("Status")
        elapsed = int(time.time() - start_time)
        logger.info(f"[{elapsed}s] App Runner status: {status}")

        if status == "RUNNING":
            logger.info("App Runner service is RUNNING and healthy!")
            return desc
        elif "FAILED" in status:
            raise RuntimeError(f"Service deployment failed with status: {status}")

        time.sleep(15)

    raise TimeoutError(f"Service did not reach RUNNING within {timeout_seconds}s")


def main():
    region = "us-east-1"
    account_id = "159142082968"
    service_name = "graph-rag-research-agent"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/graph-rag-api:latest"

    logger.info("=== Deploying GraphRAG Agent to AWS App Runner ===")
    logger.info(f"AWS Region: {region}")
    logger.info(f"Image URI:  {image_uri}")
    logger.info(f"Service:    {service_name}")

    iam_client = boto3.client("iam", region_name=region)
    apprunner_client = boto3.client("apprunner", region_name=region)

    # 1. Setup IAM Roles
    access_role_arn = setup_ecr_access_role(iam_client)
    instance_role_arn = setup_instance_role(iam_client)

    # 2. Deploy App Runner Service
    service = deploy_apprunner_service(
        apprunner_client=apprunner_client,
        service_name=service_name,
        image_uri=image_uri,
        access_role_arn=access_role_arn,
        instance_role_arn=instance_role_arn,
        region=region,
    )

    service_id = service["ServiceId"]
    service_arn = service["ServiceArn"]
    service_url = f"https://{service.get('ServiceUrl', '')}"
    status = service.get("Status", "OPERATION_IN_PROGRESS")

    logger.info("\n================ APPRUNNER DEPLOYMENT INITIATED ================")
    logger.info(f"Service ID:   {service_id}")
    logger.info(f"Service ARN:  {service_arn}")
    logger.info(f"Service URL:  {service_url}")
    logger.info(f"Status:       {status}")
    logger.info("================================================================\n")
    logger.info(f"Console link: https://{region}.console.aws.amazon.com/apprunner/home?region={region}#/services/{service_id}/service")

    # 3. Wait for service to become RUNNING
    if status != "RUNNING":
        service = wait_for_service(apprunner_client, service_arn)
        service_url = f"https://{service.get('ServiceUrl', '')}"

    logger.info("\n================ APPRUNNER SERVICE IS LIVE ================")
    logger.info(f"Service URL:  {service_url}")
    logger.info(f"Health check: {service_url}/health")
    logger.info(f"API Docs:     {service_url}/docs")
    logger.info(f"Agent Query:  {service_url}/agentcore/research")
    logger.info("===========================================================\n")


if __name__ == "__main__":
    main()
