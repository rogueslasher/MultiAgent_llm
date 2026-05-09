import os
from dataclasses import dataclass


@dataclass
class Config:
    # LLM
    llm_api_key: str
    llm_model: str
    llm_base_url: str

    # Database
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # Redis
    redis_url: str

    # API
    api_host: str
    api_port: int


def load_config() -> Config:
    missing = []

    required = [
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    for key in required:
        if not os.getenv(key):
            missing.append(key)

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your values."
        )

    return Config(
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        postgres_host=os.getenv("POSTGRES_HOST", "db"),
        postgres_port=int(os.getenv("POSTGRES_PORT", 5432)),
        postgres_db=os.getenv("POSTGRES_DB"),
        postgres_user=os.getenv("POSTGRES_USER"),
        postgres_password=os.getenv("POSTGRES_PASSWORD"),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", 8000)),
    )


config = load_config()