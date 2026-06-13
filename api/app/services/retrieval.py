from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.services.embedding import get_embedding


def search_chunks(query: str, top_k: int, session: Session) -> list[dict]:
    """Search for the most similar chunks to the query using pgvector cosine distance.

    Args:
        query: The search query text.
        top_k: The number of top results to return.
        session: The SQLAlchemy database session.

    Returns:
        A list of dicts with keys: content, document_id, filename, score.
    """
    query_embedding = get_embedding(query)

    distance_expr = Chunk.embedding.cosine_distance(query_embedding).label("distance")

    rows = (
        session.query(Chunk, Document.filename, distance_expr)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance_expr)
        .limit(top_k)
        .all()
    )

    return [
        {
            "content": chunk.content,
            "document_id": chunk.document_id,
            "filename": filename,
            "score": float(1.0 - distance),
        }
        for chunk, filename, distance in rows
    ]
