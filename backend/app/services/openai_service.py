from typing import Any
from ..core.config import settings

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - openai optional
    AsyncOpenAI = None

async def query_openai(prompt: str) -> Any:
    """Query OpenAI's chat completion API with a simple prompt."""
    if AsyncOpenAI is None:
        raise RuntimeError("openai SDK not installed")
    client = AsyncOpenAI(api_key=settings.secret_key)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
