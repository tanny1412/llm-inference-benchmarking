from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import threading

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

model = None
tokenizer = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.post("/generate")
def generate(request: GenerateRequest):
    with model_lock:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
