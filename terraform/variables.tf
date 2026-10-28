variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "agent_name" {
  description = "Bedrock Agent Name"
  type        = string
  default     = "production-graphrag-agent"
}

variable "foundation_model_id" {
  description = "Bedrock Foundation Model ID"
  type        = string
  default     = "anthropic.claude-3-5-sonnet-20241022-v2:0"
}

variable "database_url" {
  description = "PostgreSQL database URL"
  type        = string
  sensitive   = true
  default     = "postgresql://graphrag:localdevonly@127.0.0.1:5432/graphrag"
}

variable "neo4j_uri" {
  description = "Neo4j connection URI"
  type        = string
  default     = "bolt://localhost:7687"
}

