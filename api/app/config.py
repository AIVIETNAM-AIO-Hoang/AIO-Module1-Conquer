from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Args:
        None

    Returns:
        Settings instance with all config values populated.
    """

    DATABASE_URL: str
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "tinyllama"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    model_config = {"env_file": ".env"}


settings = Settings()
