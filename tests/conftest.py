import os

# Disable LangSmith network uploads during unit tests for fast hermetic execution
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
