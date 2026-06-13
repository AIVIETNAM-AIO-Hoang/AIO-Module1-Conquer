from unittest.mock import MagicMock, patch

from app.services.chunking import chunk_document


def test_chunk_document_returns_list_of_strings():
    """chunk_document returns a list of strings."""
    with patch("app.services.chunking.OllamaEmbeddings"), patch(
        "app.services.chunking.SemanticChunker"
    ) as MockChunker:
        instance = MagicMock()
        instance.split_text.return_value = ["A chunk."]
        MockChunker.return_value = instance

        result = chunk_document("A chunk.")

    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


def test_chunk_document_splits_into_multiple_chunks():
    """chunk_document returns multiple chunks for multi-paragraph input."""
    with patch("app.services.chunking.OllamaEmbeddings"), patch(
        "app.services.chunking.SemanticChunker"
    ) as MockChunker:
        instance = MagicMock()
        instance.split_text.return_value = ["First chunk.", "Second chunk."]
        MockChunker.return_value = instance

        result = chunk_document("First paragraph.\n\nSecond paragraph.")

    assert len(result) == 2
    assert result[0] == "First chunk."
    assert result[1] == "Second chunk."
