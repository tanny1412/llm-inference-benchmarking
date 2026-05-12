from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    llm = LLM(model=MODEL_NAME)
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.post("/generate")
def generate(request: GenerateRequest):
    params = SamplingParams(max_tokens=request.max_new_tokens)
    outputs = llm.generate([request.prompt], params)
    response = outputs[0].outputs[0].text
    return {"response": response}
