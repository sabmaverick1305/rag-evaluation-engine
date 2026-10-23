import os
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

class ProductionGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading local generator model ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def generate_grounded_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> GroundedRAGResponse:
        # Fallback check if no relevant chunks are present
        if not retrieved_chunks or retrieved_chunks[0].get("score", 1.0) < -2.0:
            return GroundedRAGResponse(
                answer="I cannot answer this based on the provided documents as no sufficiently relevant context was found.",
                citations=[],
                confidence_score=0.0
            )

        context_str = "\n\n".join([
            f"--- CHUNK ID: {chunk['id']} (Source: {chunk.get('source_doc', 'doc')}) ---\n{chunk['text']}" 
            for chunk in retrieved_chunks
        ])

        system_prompt = (
            "You are a precise technical AI assistant. Answer the user's question using ONLY the provided context. "
            "You MUST output valid JSON matching this exact schema:\n"
            "{\n"
            "  \"answer\": \"string\",\n"
            "  \"citations\": [\n"
            "    {\"chunk_id\": \"string\", \"source_doc\": \"string\", \"relevant_snippet\": \"string\"}\n"
            "  ],\n"
            "  \"confidence_score\": float\n"
            "}"
        )

        user_prompt = f"Context:\n{context_str}\n\nUser Question: {query}"

        # Invoke local LLM with structured instruction
        response_raw = self.llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        
        # Clean and parse JSON output into Pydantic model
        try:
            # Extract JSON block if wrapped in markdown
            if "```json" in response_raw:
                json_str = response_raw.split("```json")[1].split("```")[0].strip()
            elif "```" in response_raw:
                json_str = response_raw.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_raw.strip()

            data = json.loads(json_str)
            return GroundedRAGResponse(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Fallback parser safety net if formatting fails
            return GroundedRAGResponse(
                answer=response_raw.strip(),
                citations=[
                    Citation(
                        chunk_id=retrieved_chunks[0]["id"],
                        source_doc=retrieved_chunks[0].get("source_doc", "unknown"),
                        relevant_snippet=retrieved_chunks[0]["text"][:150]
                    )
                ],
                confidence_score=0.5
            )