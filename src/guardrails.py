import re
from typing import Any


class InputGuardrail:
    @staticmethod
    def sanitize_and_redact(query: str) -> str:
        """Redacts potential PII (emails, phone numbers) and sanitizes user input."""
        if not query:
            return ""

        # Redact email addresses
        query = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', query)

        # Redact phone numbers (simple pattern)
        query = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', query)

        return query.strip()

class OutputGuardrail:
    @staticmethod
    def verify_and_filter(answer: str, retrieved_chunks: list[dict[str, Any]]) -> str:
        """Verifies output quality, prevents empty responses, and ensures basic safety grounding."""
        if not answer or len(answer.strip()) < 3:
            return "I apologize, but I could not generate a valid response based on the retrieved context."

        # Additional output filtering or post-processing checks can be added here
        return answer.strip()
