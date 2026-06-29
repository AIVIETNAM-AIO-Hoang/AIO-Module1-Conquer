"""HTTP client wrappers for the RAG API."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def check_health() -> bool:
    """Check whether the API is reachable.
 c
    Pings a lightweight endpoint that does not depend on the database, so it
    reflects API reachability rather than database state.
 
    Returns:
        True if the API responds, False if it cannot be reached.
    """
    try:
        response = requests.get(f"{_BASE_URL}/openapi.json", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def upload_document(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """Upload a document for chunking and embedding.

    Args:
        file_bytes: Raw file content.
        filename: Original filename including extension.
        content_type: MIME type of the file.

    Returns:
        Dict with document_id, filename, and chunk_count.

    Raises:
        requests.HTTPError: If the API returns an error status.
    """
    response = requests.post(
        f"{_BASE_URL}/api/documents/upload",
        files={"file": (filename, file_bytes, content_type)},
    )
    response.raise_for_status()
    return response.json()


def list_documents() -> list[dict]:
    """Retrieve all uploaded documents with chunk counts.

    Returns:
        List of dicts with document_id, filename, chunk_count, and created_at.

    Raises:
        requests.HTTPError: If the API returns an error status.
    """
    response = requests.get(f"{_BASE_URL}/api/documents/list")
    response.raise_for_status()
    return response.json()["documents"]


def remove_document(document_id: str) -> dict:
    """Delete a document and all its chunks.

    Args:
        document_id: UUID string of the document to delete.

    Returns:
        Dict with document_id and deleted flag.

    Raises:
        requests.HTTPError: If the API returns an error status.
    """
    response = requests.delete(
        f"{_BASE_URL}/api/documents/remove",
        params={"document_id": document_id},
    )
    response.raise_for_status()
    return response.json()


def prompt(query: str, top_k: int = 5) -> dict:
    """Send a query and get a grounded answer with source citations.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve for context.

    Returns:
        Dict with answer string and sources list.

    Raises:
        requests.HTTPError: If the API returns an error status.
    """
    response = requests.post(
        f"{_BASE_URL}/api/rag/prompt",
        json={"query": query, "top_k": top_k},
    )
    response.raise_for_status()
    return response.json()
