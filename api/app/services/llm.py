import requests

from app.config import settings


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Generate a grounded answer using the local Ollama LLM.

    Args:
        query: The user's question.
        context_chunks: Retrieved text chunks that ground the answer.

    Returns:
        The generated answer string.

    Raises:
        requests.HTTPError: If the Ollama API call fails.
    """
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
    prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the provided context. "
        "If the answer is not found in the context, say so explicitly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json={"model": settings.OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]
