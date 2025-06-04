from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.openai_service import query_openai

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

class Prompt(BaseModel):
    prompt: str

@router.post("/llm/openai")
async def openai_endpoint(data: Prompt):
    try:
        text = await query_openai(data.prompt)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="OpenAI service unavailable")
    return {"response": text}
