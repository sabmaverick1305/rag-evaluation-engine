---
name: deploy-bedrock
description: Operational workflow to provision, update, and verify AWS Bedrock Agent Core and Action Groups.
---

# Deploy Bedrock Agent Core Workflow

Use this skill when you need to deploy, update, or verify the AWS Bedrock Agent, Lambda Action Group, or foundation models in AWS.

## Prerequisites
1. AWS CLI configured with appropriate credentials or active role:
   ```bash
   aws sts get-caller-identity
   ```
2. Python dependencies installed:
   ```bash
   pip install boto3
   ```

## Procedure

### Step 1: Dry Run Verification
Validate configuration and parameters without making modifications:
```bash
python scripts/deploy_bedrock_agent.py --dry-run
```

### Step 2: Execute Deployment
Deploy IAM execution roles, Lambda Action Group, Bedrock Agent, and the `production` alias:
```bash
python scripts/deploy_bedrock_agent.py \
  --region us-east-1 \
  --model-id "anthropic.claude-3-5-sonnet-20241022-v2:0" \
  --agent-name "rag-graph-research-agent" \
  --alias-name "production"
```

### Step 3: Configure Environment Variables
Copy the script outputs to your `.env` or deployment secrets:
```bash
AWS_REGION=us-east-1
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
AGENT_MODE=bedrock_agent
BEDROCK_AGENT_ID=<Agent-Id-From-Output>
BEDROCK_AGENT_ALIAS_ID=<Alias-Id-From-Output>
```

### Step 4: Verification & Smoke Test
Verify the deployment by querying the `/health` endpoint:
```bash
curl -X GET http://localhost:8000/health
```
Execute a test research query:
```bash
curl -X POST http://localhost:8000/bedrock-agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain how multi-head attention functions in transformer models."}'
```

