terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# IAM Roles
# ---------------------------------------------------------------------------
resource "aws_iam_role" "bedrock_agent_role" {
  name = "${var.agent_name}-bedrock-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "bedrock.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_agent_policy" {
  name = "BedrockModelInvocation"
  role = aws_iam_role.bedrock_agent_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetFoundationModel"
      ]
      Effect   = "Allow"
      Resource = "*"
    }]
  })
}

resource "aws_iam_role" "lambda_role" {
  name = "${var.agent_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# Action Group Lambda Function
# ---------------------------------------------------------------------------
data "archive_file" "lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/build/action_group.zip"
  source_dir  = "${path.module}/../src"
}

resource "aws_lambda_function" "action_group" {
  function_name    = "${var.agent_name}-action-group"
  role             = aws_iam_role.lambda_role.arn
  handler          = "aws.lambda_action_group.lambda_handler"
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 512
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      DATABASE_URL = var.database_url
      NEO4J_URI    = var.neo4j_uri
    }
  }
}

# ---------------------------------------------------------------------------
# Bedrock Agent & Action Group
# ---------------------------------------------------------------------------
resource "aws_bedrockagent_agent" "graphrag_agent" {
  agent_name                  = var.agent_name
  agent_resource_role_arn     = aws_iam_role.bedrock_agent_role.arn
  foundation_model            = var.foundation_model_id
  idle_session_ttl_in_seconds = 1800
  instruction = "You are an expert GraphRAG Deep Research Agent. Answer queries by retrieving context using the GraphRAGActionGroup tools. Always ground your technical answers in the retrieved chunks and provide citations. Output responses formatted as JSON matching GroundedRAGResponse: {\"answer\": \"...\", \"citations\": [{\"chunk_id\": \"...\", \"source_doc\": \"...\", \"relevant_snippet\": \"...\"}], \"confidence_score\": 0.9}."
}

resource "aws_lambda_permission" "allow_bedrock" {
  statement_id  = "AllowBedrockInvocation"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.action_group.function_name
  principal     = "bedrock.amazonaws.com"
  source_arn    = aws_bedrockagent_agent.graphrag_agent.agent_arn
}

resource "aws_bedrockagent_agent_action_group" "action_group" {
  agent_id                 = aws_bedrockagent_agent.graphrag_agent.id
  agent_version            = "DRAFT"
  action_group_name        = "GraphRAGActionGroup"
  description              = "Tools for pgvector hybrid retrieval and Neo4j ontology traversal"
  action_group_state       = "ENABLED"

  action_group_executor {
    lambda = aws_lambda_function.action_group.arn
  }

  api_schema {
    payload = file("${path.module}/../src/aws/action_group_openapi.json")
  }

  depends_on = [aws_lambda_permission.allow_bedrock]
}

resource "aws_bedrockagent_agent_alias" "prod_alias" {
  agent_id         = aws_bedrockagent_agent.graphrag_agent.id
  agent_alias_name = "production"
  description      = "Production alias for GraphRAG Agent"

  depends_on = [aws_bedrockagent_agent_action_group.action_group]
}

