import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services.retrieval import search_chunks


def _make_row(content: str, filename: str, distance: float):
    """Build a mock query result row."""
    chunk = MagicMock()
    chunk.content = content
    chunk.document_id = uuid4()
    return (chunk, filename, distance)


def test_search_chunks_returns_correct_shape():
    """search_chunks returns a list of dicts with the expected keys."""
    mock_session = MagicMock()
    mock_q = MagicMock()
    mock_session.query.return_value = mock_q
    mock_q.join.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.limit.return_value = mock_q
    mock_q.all.return_value = [_make_row("chunk text", "doc.txt", 0.2)]

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768):
        results = search_chunks("query", 1, mock_session)

    assert len(results) == 1
    assert set(results[0].keys()) == {"content", "document_id", "filename", "score"}


def test_search_chunks_converts_distance_to_score():
    """score is 1 - cosine_distance so higher means more similar."""
    mock_session = MagicMock()
    mock_q = MagicMock()
    mock_session.query.return_value = mock_q
    mock_q.join.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.limit.return_value = mock_q
    mock_q.all.return_value = [
        _make_row("best", "a.txt", 0.1),
        _make_row("worse", "b.txt", 0.4),
    ]

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768):
        results = search_chunks("query", 2, mock_session)

    assert results[0]["score"] == pytest.approx(0.9)
    assert results[1]["score"] == pytest.approx(0.6)
