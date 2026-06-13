"""HTTP client for the chatbotrag API."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Upload a document to the API for chunking and embedding.

    Args:
        file_bytes: Raw file content.
        filename: Original filename, used to determine content type.

    Returns:
        Dict with document_id, filename, and chunk_count.

    Raises:
        requests.HTTPError: If the server returns a non-2xx response.
    """
    response = requests.post(
        f"{_BASE_URL}/api/documents/upload",
        files={"file": (filename, file_bytes)},
    )
    response.raise_for_status()
    return response.json()


def list_documents() -> list[dict]:
    """Fetch all documents from the API.

    Returns:
        List of dicts with document_id, filename, chunk_count, created_at.

    Raises:
        requests.HTTPError: If the server returns a non-2xx response.
    """
    response = requests.get(f"{_BASE_URL}/api/documents/list")
    response.raise_for_status()
    return response.json()["documents"]


def delete_document(document_id: str) -> dict:
    """Delete a document and its chunks from the API.

    Args:
        document_id: UUID string of the document to delete.

    Returns:
        Dict with document_id and deleted flag.

    Raises:
        requests.HTTPError: If the server returns a non-2xx response.
    """
    response = requests.delete(
        f"{_BASE_URL}/api/documents/remove",
        params={"document_id": document_id},
    )
    response.raise_for_status()
    return response.json()
