from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas import (
    ChunkResult,
    PromptRequest,
    PromptResponse,
    RetrieveRequest,
    RetrieveResponse,
    Source,
)
from app.services.llm import generate_answer
from app.services.retrieval import search_chunks

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest, session: Session = Depends(get_session)):
    """Perform dense similarity search and return ranked chunks.

    Args:
        request: Query and top_k parameters.
        session: Database session.

    Returns:
        RetrieveResponse with a list of ranked ChunkResult items.
    """
    chunks = search_chunks(request.query, request.top_k, session)
    return RetrieveResponse(
        chunks=[
            ChunkResult(
                content=c["content"],
                document_id=c["document_id"],
                filename=c["filename"],
                score=c["score"],
            )
            for c in chunks
        ]
    )


@router.post("/prompt", response_model=PromptResponse)
def prompt(request: PromptRequest, session: Session = Depends(get_session)):
    """Retrieve relevant chunks and generate a grounded answer.

    Args:
        request: Query and top_k parameters.
        session: Database session.

    Returns:
        PromptResponse with answer text and source attribution.

    Raises:
        HTTPException: 404 if no relevant chunks are found.
    """
    chunks = search_chunks(request.query, request.top_k, session)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    answer = generate_answer(request.query, [c["content"] for c in chunks])
    return PromptResponse(
        answer=answer,
        sources=[
            Source(
                document_id=c["document_id"],
                filename=c["filename"],
                content=c["content"],
                score=c["score"],
            )
            for c in chunks
        ],
    )
