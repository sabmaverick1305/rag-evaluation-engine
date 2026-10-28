output "bedrock_agent_id" {
  description = "The ID of the provisioned AWS Bedrock Agent"
  value       = aws_bedrockagent_agent.graphrag_agent.id
}

output "bedrock_agent_alias_id" {
  description = "The ID of the Bedrock Agent production alias"
  value       = aws_bedrockagent_agent_alias.prod_alias.agent_alias_id
}

output "action_group_lambda_arn" {
  description = "ARN of the GraphRAG Action Group Lambda function"
  value       = aws_lambda_function.action_group.arn
}

