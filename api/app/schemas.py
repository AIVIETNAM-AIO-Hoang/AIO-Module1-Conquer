from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Source(BaseModel):
    """A source document chunk cited in a RAG answer."""

    document_id: UUID
    filename: str
    content: str
    score: float | None = None   # <-- thêm dòng này


class UploadResponse(BaseModel):
    """Response body for POST /api/documents/upload."""

    document_id: UUID
    filename: str
    chunk_count: int


class DocumentInfo(BaseModel):
    """Summary of a single document returned by the list endpoint."""

    document_id: UUID
    filename: str
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Response body for GET /api/documents/list."""

    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    """Response body for DELETE /api/documents/remove."""

    document_id: UUID
    deleted: bool


class RetrieveRequest(BaseModel):
    """Request body for POST /api/rag/retrieve."""

    query: str
    top_k: int = 5


class ChunkResult(BaseModel):
    """A single ranked chunk returned by retrieval."""

    content: str
    document_id: UUID
    filename: str
    score: float


class RetrieveResponse(BaseModel):
    """Response body for POST /api/rag/retrieve."""

    chunks: list[ChunkResult]


class PromptRequest(BaseModel):
    """Request body for POST /api/rag/prompt."""

    query: str
    top_k: int = 5


class Source(BaseModel):
    """A source document chunk cited in a RAG answer."""

    document_id: UUID
    filename: str
    content: str


class PromptResponse(BaseModel):
    """Response body for POST /api/rag/prompt."""

    answer: str
    sources: list[Source]
