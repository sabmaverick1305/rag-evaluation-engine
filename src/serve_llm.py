# Install requirements via: uv pip install vllm fastapi uvicorn
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI(title="Optimized Graph RAG Inference Service", version="2.0.0")

# Initialize vLLM engine with our fine-tuned QLoRA/merged adapter model
# vLLM optimizes memory via PagedAttention and handles continuous batching natively.
MODEL_PATH = os.getenv("MODEL_PATH", "./qwen-graphrag-adapter-final")

print(f"Loading model into vLLM engine from {MODEL_PATH}...")
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=2048
)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.1

@app.post("/generate")
async def optimize_and_generate(request: InferenceRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=["<|im_end|>"]
        )

        # vLLM automatically executes continuous batching and high-throughput decoding
        outputs = llm.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        return {
            "response": generated_text,
            "metrics": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "output_tokens": len(outputs[0].outputs[0].token_ids)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
