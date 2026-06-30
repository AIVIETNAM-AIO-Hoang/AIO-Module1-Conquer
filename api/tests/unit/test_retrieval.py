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


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases: empty query and top_k boundaries (added)
# ──────────────────────────────────────────────────────────────────────────────


def _mock_session_returning(rows):
    """Build a mock session whose query chain resolves to `rows`.

    Returns a tuple (session, query_mock) so callers can assert on the chain,
    e.g. that ``.limit`` was called with a specific top_k.
    """
    session = MagicMock()
    q = MagicMock()
    session.query.return_value = q
    q.join.return_value = q
    q.order_by.return_value = q
    q.limit.return_value = q
    q.all.return_value = rows
    return session, q


def test_search_chunks_empty_query_is_still_embedded_and_searched():
    """An empty query is not rejected here; it is embedded and searched.

    Documents the absence of input validation at the service layer — useful to
    pin down before deciding whether to enforce a non-empty query upstream.
    """
    session, _ = _mock_session_returning([_make_row("hit", "doc.txt", 0.2)])

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768) as embed:
        results = search_chunks("", 5, session)

    embed.assert_called_once_with("")
    assert len(results) == 1


def test_search_chunks_passes_top_k_through_to_limit():
    """The requested top_k is forwarded verbatim to the SQL LIMIT clause."""
    session, q = _mock_session_returning([])

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768):
        search_chunks("query", 3, session)

    q.limit.assert_called_once_with(3)


def test_search_chunks_top_k_zero_returns_empty_list():
    """top_k=0 yields no rows and an empty result list."""
    session, q = _mock_session_returning([])

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768):
        results = search_chunks("query", 0, session)

    q.limit.assert_called_once_with(0)
    assert results == []


def test_search_chunks_negative_top_k_is_not_validated():
    """A negative top_k is forwarded unchanged (no guard at this layer).

    Documents current behavior: validation, if desired, belongs upstream
    (schema/endpoint), since a negative LIMIT would error at the database.
    """
    session, q = _mock_session_returning([])

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768):
        search_chunks("query", -1, session)

    q.limit.assert_called_once_with(-1)


def test_search_chunks_returns_all_available_when_top_k_exceeds_rows():
    """When fewer rows exist than top_k, all available rows are returned."""
    rows = [_make_row("a", "a.txt", 0.1), _make_row("b", "b.txt", 0.3)]
    session, _ = _mock_session_returning(rows)

    with patch("app.services.retrieval.get_embedding", return_value=[0.1] * 768):
        results = search_chunks("query", 100, session)

    assert len(results) == 2
