import requests

from app.config import settings


def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text using Ollama.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        requests.HTTPError: If the Ollama API call fails.
    """
    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": settings.OLLAMA_EMBEDDING_MODEL, "prompt": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]
