from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

from app.config import settings


def chunk_document(text: str) -> list[str]:
    """Split text into semantic chunks using LangChain SemanticChunker.

    Args:
        text: The document text to chunk.

    Returns:
        A list of text chunk strings.
    """
    embeddings = OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_EMBEDDING_MODEL,
    )
    chunker = SemanticChunker(embeddings)
    return chunker.split_text(text)
