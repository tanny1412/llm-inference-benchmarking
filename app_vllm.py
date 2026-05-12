from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import uuid

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine_args = AsyncEngineArgs(model=MODEL_NAME)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.post("/generate")
async def generate(request: GenerateRequest):
    params = SamplingParams(max_tokens=request.max_new_tokens)
    request_id = str(uuid.uuid4())
    async for output in engine.generate(request.prompt, params, request_id):
        final_output = output
    response = final_output.outputs[0].text
    return {"response": response}
