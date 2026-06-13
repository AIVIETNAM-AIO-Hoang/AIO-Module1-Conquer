import io
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import Chunk, Document
from app.schemas import DeleteResponse, DocumentInfo, DocumentListResponse, UploadResponse
from app.services.chunking import chunk_document
from app.services.embedding import get_embedding

router = APIRouter(prefix="/api/documents", tags=["documents"])

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
_CONTENT_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def _resolve_content_type(filename: str | None, declared: str | None) -> str:
    """Determine content type from file extension, falling back to the declared type.

    Args:
        filename: The uploaded filename.
        declared: The content_type declared by the client.

    Returns:
        A resolved content type string.

    Raises:
        HTTPException: If the file extension is not supported.
    """
    if filename:
        for ext in _SUPPORTED_EXTENSIONS:
            if filename.lower().endswith(ext):
                return _CONTENT_TYPE_MAP[ext]
    if declared in ("application/pdf", "text/plain", "text/markdown"):
        return declared
    raise HTTPException(status_code=422, detail="Unsupported file type. Use .pdf, .txt, or .md")


def _extract_text(content: bytes, content_type: str) -> str:
    """Extract plain text from file bytes according to content type.

    Args:
        content: Raw file bytes.
        content_type: MIME type of the file.

    Returns:
        Extracted plain text string.
    """
    if content_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="replace")


@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Parse, chunk, embed, and persist an uploaded document.

    Args:
        file: The uploaded file (pdf, txt, or md).
        session: Database session.

    Returns:
        UploadResponse with document_id, filename, and chunk_count.

    Raises:
        HTTPException: 422 if the file type is not supported.
    """
    content_type = _resolve_content_type(file.filename, file.content_type)
    raw = file.file.read()
    text = _extract_text(raw, content_type)

    chunks = chunk_document(text)

    document = Document(filename=file.filename or "upload", content_type=content_type)
    session.add(document)
    session.flush()

    for i, chunk_text in enumerate(chunks):
        embedding = get_embedding(chunk_text)
        session.add(
            Chunk(
                document_id=document.id,
                content=chunk_text,
                embedding=embedding,
                chunk_index=i,
            )
        )

    session.commit()
    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        chunk_count=len(chunks),
    )


@router.get("/list", response_model=DocumentListResponse)
def list_documents(session: Session = Depends(get_session)):
    """Return all documents with their chunk counts.

    Args:
        session: Database session.

    Returns:
        DocumentListResponse containing a list of DocumentInfo items.
    """
    rows = (
        session.query(Document, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk, Document.id == Chunk.document_id)
        .group_by(Document.id)
        .all()
    )
    documents = [
        DocumentInfo(
            document_id=doc.id,
            filename=doc.filename,
            chunk_count=count,
            created_at=doc.created_at,
        )
        for doc, count in rows
    ]
    return DocumentListResponse(documents=documents)


@router.delete("/remove", response_model=DeleteResponse)
def remove_document(document_id: UUID, session: Session = Depends(get_session)):
    """Delete a document and all its chunks.

    Args:
        document_id: UUID of the document to delete.
        session: Database session.

    Returns:
        DeleteResponse confirming deletion.

    Raises:
        HTTPException: 404 if the document does not exist.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    session.delete(document)
    session.commit()
    return DeleteResponse(document_id=document_id, deleted=True)
