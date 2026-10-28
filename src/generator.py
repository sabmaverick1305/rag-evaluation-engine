import json
import os
import sys
from typing import Any

from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.schemas import GroundedRAGResponse


class ProductionGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", adapter_path: str | None = "./qwen-graphrag-adapter-final", *, do_sample: bool = True, max_new_tokens: int = 512):
        if sys.platform == "darwin":
            # Concurrent safetensors loading can deadlock with this local stack.
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        print(f"Loading local generator model ({model_id})...")
        print("Loading base model and fine-tuned LoRA adapter..." if adapter_path else "Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Tokenizer loaded; loading base model weights...")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
        self.model.eval()
        print("LoRA adapter loaded." if adapter_path else "Base model loaded without an adapter.")

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_full_text=False,
            **({"temperature": 0.1} if do_sample else {}),
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def generate_grounded_answer(self, query: str, retrieved_chunks: list[dict[str, Any]], *, diagnostics: dict | None = None) -> GroundedRAGResponse:
        if diagnostics is not None:
            diagnostics.update(raw_output=None, schema_valid=False, generation_skipped=False, validation_failure=None)
        # Fallback check if no relevant chunks are present
        if not retrieved_chunks or retrieved_chunks[0].get("score", 1.0) < -2.0:
            if diagnostics is not None:
                diagnostics["generation_skipped"] = True
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
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response_raw = self.llm.invoke(prompt)
        if diagnostics is not None:
            diagnostics["raw_output"] = response_raw

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
            response = GroundedRAGResponse(**data)
            if diagnostics is not None:
                diagnostics["schema_valid"] = True
            return response
        except (json.JSONDecodeError, ValidationError) as e:
            if diagnostics is not None:
                diagnostics["validation_failure"] = {"type": type(e).__name__, "message": str(e)}
            # Fallback parser safety net if formatting fails
            return GroundedRAGResponse(
                answer=response_raw.strip(),
                # Parsing failure is not evidence of grounding: never invent a citation.
                citations=[],
                confidence_score=0.0
            )
