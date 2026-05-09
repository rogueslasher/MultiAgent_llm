from openai import AsyncOpenAI
import os

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

def get_model() -> str:
    return os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")