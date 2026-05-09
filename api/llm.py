from openai import AsyncOpenAI
from api.config import config


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )


def get_model() -> str:
    return config.llm_model