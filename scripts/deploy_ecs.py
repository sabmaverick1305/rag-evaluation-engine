#!/usr/bin/env python3
"""Automated Deployment Script for AWS ECS Fargate.

Provisions:
1. ECS Task Execution Role & Task Runtime Role (with Bedrock + Lambda permissions)
2. CloudWatch Log Group for container logs
3. Security Group allowing HTTP ingress on port 8000
4. ECS Fargate Task Definition using ECR image `graph-rag-api:latest`
5. ECS Service on cluster `graph-rag-cluster` with public IP assignment
6. Fetches public IP and verifies health endpoint
"""

import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy_ecs")


def setup_execution_role(iam_client: Any) -> str:
    """Creates or gets the ECS Task Execution IAM Role for pulling ECR images and logging."""
    role_name = "ECSTaskExecutionRole"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        logger.info(f"Using existing ECS execution role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"Creating ECS execution role: {role_name}...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role allowing ECS agent to pull from ECR and send logs to CloudWatch",
        )
        role_arn = response["Role"]["Arn"]

    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )
    return role_arn


def setup_task_role(iam_client: Any) -> str:
    """Creates or gets the runtime IAM role assumed by the application inside the container."""
    role_name = "ECSTaskRuntimeRole"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        logger.info(f"Using existing ECS task runtime role: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"Creating ECS task runtime role: {role_name}...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Runtime role for application calling Bedrock and Lambda",
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
        PolicyName="ECSApplicationBedrockAndLambdaAccess",
        PolicyDocument=json.dumps(policy_doc),
    )
    return role_arn


def setup_log_group(logs_client: Any, log_group_name: str) -> None:
    """Ensures CloudWatch Log Group exists."""
    try:
        logs_client.create_log_group(logGroupName=log_group_name)
        logger.info(f"Created CloudWatch Log Group: {log_group_name}")
    except logs_client.exceptions.ResourceAlreadyExistsException:
        logger.info(f"Log Group already exists: {log_group_name}")


def setup_security_group_and_subnets(ec2_client: Any) -> Tuple[str, List[str]]:
    """Gets default VPC, creates security group allowing port 8000, and returns public subnets."""
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError("No default VPC found in this region.")
    vpc_id = vpcs[0]["VpcId"]

    subnets_res = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_ids = [s["SubnetId"] for s in subnets_res["Subnets"]]
    if not subnet_ids:
        raise RuntimeError(f"No subnets found for VPC {vpc_id}")

    sg_name = "graph-rag-ecs-sg"
    try:
        sgs = ec2_client.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "group-name", "Values": [sg_name]},
            ]
        )["SecurityGroups"]
        if sgs:
            sg_id = sgs[0]["GroupId"]
            logger.info(f"Using existing security group: {sg_id}")
            return sg_id, subnet_ids
    except Exception as e:
        logger.warning(f"Error checking security groups: {e}")

    logger.info(f"Creating security group {sg_name} in VPC {vpc_id}...")
    res = ec2_client.create_security_group(
        GroupName=sg_name,
        Description="Security group for GraphRAG ECS Fargate service",
        VpcId=vpc_id,
    )
    sg_id = res["GroupId"]

    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 8000,
                "ToPort": 8000,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "FastAPI HTTP Access"}],
            }
        ],
    )
    logger.info(f"Configured security group {sg_id} for port 8000 ingress.")
    return sg_id, subnet_ids


def register_task_definition(
    ecs_client: Any,
    family: str,
    image_uri: str,
    execution_role_arn: str,
    task_role_arn: str,
    log_group_name: str,
    region: str,
) -> str:
    """Registers an ECS Fargate task definition."""
    container_def = {
        "name": "graph-rag-api",
        "image": image_uri,
        "essential": True,
        "portMappings": [
            {
                "containerPort": 8000,
                "hostPort": 8000,
                "protocol": "tcp",
            }
        ],
        "environment": [
            {"name": "AWS_REGION", "value": region},
            {"name": "AWS_DEFAULT_REGION", "value": region},
            {"name": "BEDROCK_MODEL_ID", "value": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
            {"name": "LLM_PROVIDER", "value": "bedrock"},
            {"name": "AGENT_MODE", "value": "agentcore"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group_name,
                "awslogs-region": region,
                "awslogs-stream-prefix": "ecs",
            },
        },
    }

    response = ecs_client.register_task_definition(
        family=family,
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu="1024",
        memory="2048",
        executionRoleArn=execution_role_arn,
        taskRoleArn=task_role_arn,
        containerDefinitions=[container_def],
    )
    task_def_arn = response["taskDefinition"]["taskDefinitionArn"]
    logger.info(f"Registered Task Definition: {task_def_arn}")
    return task_def_arn


def deploy_or_update_service(
    ecs_client: Any,
    cluster_name: str,
    service_name: str,
    task_def_arn: str,
    subnet_ids: List[str],
    sg_id: str,
) -> str:
    """Creates or updates the ECS Fargate service."""
    network_config = {
        "awsvpcConfiguration": {
            "subnets": subnet_ids[:2],  # Use first 2 subnets
            "securityGroups": [sg_id],
            "assignPublicIp": "ENABLED",
        }
    }

    try:
        desc = ecs_client.describe_services(cluster=cluster_name, services=[service_name])
        services = desc.get("services", [])
        if services and services[0].get("status") == "ACTIVE":
            logger.info(f"Updating existing ECS service: {service_name}...")
            res = ecs_client.update_service(
                cluster=cluster_name,
                service=service_name,
                taskDefinition=task_def_arn,
                desiredCount=1,
                networkConfiguration=network_config,
                forceNewDeployment=True,
            )
            return res["service"]["serviceArn"]
    except ClientError as e:
        logger.warning(f"Error checking service: {e}")

    logger.info(f"Creating new ECS Fargate service: {service_name}...")
    # Add slight delay for IAM propagation
    time.sleep(5)
    for attempt in range(6):
        try:
            res = ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                launchType="FARGATE",
                desiredCount=1,
                networkConfiguration=network_config,
            )
            return res["service"]["serviceArn"]
        except ClientError as e:
            if "role" in str(e).lower() or "unable to assume" in str(e).lower():
                logger.info(f"Waiting for IAM role propagation (attempt {attempt+1}/6)...")
                time.sleep(5)
            else:
                raise

    raise RuntimeError("Failed to create ECS service after retries")


def wait_for_task_ip(
    ecs_client: Any,
    ec2_client: Any,
    cluster_name: str,
    service_name: str,
    timeout_seconds: int = 300,
) -> str:
    """Polls ECS tasks until one is RUNNING and extracts its public IP."""
    logger.info("Waiting for ECS Fargate task to reach RUNNING state and get public IP...")
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        tasks = ecs_client.list_tasks(cluster=cluster_name, serviceName=service_name)["taskArns"]
        if tasks:
            task_desc = ecs_client.describe_tasks(cluster=cluster_name, tasks=[tasks[0]])["tasks"][0]
            status = task_desc.get("lastStatus")
            elapsed = int(time.time() - start_time)
            logger.info(f"[{elapsed}s] Task status: {status}")

            if status == "RUNNING":
                # Extract ENI
                for attachment in task_desc.get("attachments", []):
                    if attachment.get("type") == "ElasticNetworkInterface":
                        for detail in attachment.get("details", []):
                            if detail.get("name") == "networkInterfaceId":
                                eni_id = detail.get("value")
                                eni_desc = ec2_client.describe_network_interfaces(
                                    NetworkInterfaceIds=[eni_id]
                                )["NetworkInterfaces"][0]
                                public_ip = eni_desc.get("Association", {}).get("PublicIp")
                                if public_ip:
                                    logger.info(f"Found Public IP: {public_ip}")
                                    return public_ip

        time.sleep(10)

    raise TimeoutError(f"Task did not reach RUNNING with public IP within {timeout_seconds}s")


def main():
    region = "us-east-1"
    account_id = "159142082968"
    cluster_name = "graph-rag-cluster"
    service_name = "graph-rag-research-agent"
    task_family = "graph-rag-research-agent"
    log_group_name = "/ecs/graph-rag-research-agent"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/graph-rag-api:latest"

    logger.info("=== Deploying GraphRAG Agent to AWS ECS Fargate ===")
    logger.info(f"AWS Region: {region}")
    logger.info(f"Cluster:    {cluster_name}")
    logger.info(f"Service:    {service_name}")
    logger.info(f"Image URI:  {image_uri}")

    iam_client = boto3.client("iam", region_name=region)
    ecs_client = boto3.client("ecs", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)
    logs_client = boto3.client("logs", region_name=region)

    # 1. IAM Roles
    execution_role_arn = setup_execution_role(iam_client)
    task_role_arn = setup_task_role(iam_client)

    # 2. CloudWatch Logs
    setup_log_group(logs_client, log_group_name)

    # 3. Security Group & Subnets
    sg_id, subnet_ids = setup_security_group_and_subnets(ec2_client)

    # 4. Task Definition
    task_def_arn = register_task_definition(
        ecs_client=ecs_client,
        family=task_family,
        image_uri=image_uri,
        execution_role_arn=execution_role_arn,
        task_role_arn=task_role_arn,
        log_group_name=log_group_name,
        region=region,
    )

    # 5. Service
    service_arn = deploy_or_update_service(
        ecs_client=ecs_client,
        cluster_name=cluster_name,
        service_name=service_name,
        task_def_arn=task_def_arn,
        subnet_ids=subnet_ids,
        sg_id=sg_id,
    )

    # 6. Wait for RUNNING and Public IP
    public_ip = wait_for_task_ip(ecs_client, ec2_client, cluster_name, service_name)

    service_url = f"http://{public_ip}:8000"
    console_url = f"https://{region}.console.aws.amazon.com/ecs/v2/clusters/{cluster_name}/services/{service_name}?region={region}"

    logger.info("\n================ AWS ECS FARGATE SERVICE LIVE ================")
    logger.info(f"Public IP:    {public_ip}")
    logger.info(f"Swagger Docs: {service_url}/docs")
    logger.info(f"Health Check: {service_url}/health")
    logger.info(f"Agent URL:    {service_url}/agentcore/research")
    logger.info(f"AWS Console:  {console_url}")
    logger.info("=============================================================\n")

    # Quick verification
    logger.info("Verifying health endpoint...")
    for _ in range(10):
        try:
            req = urllib.request.Request(f"{service_url}/health")
            with urllib.request.urlopen(req, timeout=5) as res:
                if res.status == 200:
                    logger.info(f"Health check succeeded! Response: {res.read().decode('utf-8')}")
                    break
        except Exception as e:
            logger.info(f"Waiting for container server to accept connections ({e})...")
            time.sleep(5)


if __name__ == "__main__":
    main()
