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

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Fallback Check: If no chunks retrieved or confidence score is below threshold
        if not retrieved_chunks or retrieved_chunks[0].get("score", 1.0) < -2.0:
            return {
                "query": query,
                "answer": "I cannot answer this based on the provided documents as no sufficiently relevant context was found.",
                "sources": []
            }

        context_str = "\n\n".join([f"Source [{chunk.get('source_doc', 'doc')}]: {chunk['text']}" for chunk in retrieved_chunks])
        
        prompt = f"""You are a precise technical AI assistant. Answer the user's question using ONLY the provided context below. If the answer cannot be found in the context, state "I cannot answer this based on the provided documents." Do not hallucinate or extrapolate.

Context:
{context_str}

User Question: {query}

Answer:"""

        response = self.llm.invoke(prompt)
        
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response.strip()

        return {
            "query": query,
            "answer": answer,
            "sources": [chunk["id"] for chunk in retrieved_chunks]
        }