
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str = Field(description="The exact chunk ID used as evidence (e.g., ai_engineering_ch1_chunk_333).")
    source_doc: str = Field(description="The source document identifier.")
    relevant_snippet: str = Field(description="A direct quote or paraphrased fact from the chunk supporting the answer.")

class GroundedRAGResponse(BaseModel):
    answer: str = Field(description="The precise technical answer grounded exclusively in the provided context.")
    citations: list[Citation] = Field(description="List of exact chunk citations supporting the claims made in the answer.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Self-reported confidence between 0.0 and 1.0; not a correctness measurement.")
