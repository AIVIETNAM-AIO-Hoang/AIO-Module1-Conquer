"""Unit tests for api_client — all HTTP calls are mocked."""

from unittest.mock import MagicMock, patch

import pytest

import api_client


def _mock_response(json_data: dict | list, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


class TestUploadDocument:
    def test_returns_upload_response(self):
        payload = {"document_id": "abc-123", "filename": "doc.pdf", "chunk_count": 5}
        with patch("api_client.requests.post", return_value=_mock_response(payload)) as mock_post:
            result = api_client.upload_document(b"bytes", "doc.pdf")
        assert result == payload
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "files" in kwargs

    def test_raises_on_http_error(self):
        mock = _mock_response({}, status_code=422)
        mock.raise_for_status.side_effect = Exception("422 Unprocessable Entity")
        with patch("api_client.requests.post", return_value=mock):
            with pytest.raises(Exception, match="422"):
                api_client.upload_document(b"bytes", "bad.xyz")


class TestListDocuments:
    def test_returns_documents_list(self):
        documents = [
            {"document_id": "abc", "filename": "a.txt", "chunk_count": 3, "created_at": "2026-01-01T00:00:00"}
        ]
        payload = {"documents": documents}
        with patch("api_client.requests.get", return_value=_mock_response(payload)):
            result = api_client.list_documents()
        assert result == documents

    def test_raises_on_http_error(self):
        mock = _mock_response({}, status_code=500)
        mock.raise_for_status.side_effect = Exception("500 Server Error")
        with patch("api_client.requests.get", return_value=mock):
            with pytest.raises(Exception, match="500"):
                api_client.list_documents()


class TestDeleteDocument:
    def test_returns_delete_response(self):
        payload = {"document_id": "abc-123", "deleted": True}
        with patch("api_client.requests.delete", return_value=_mock_response(payload)) as mock_delete:
            result = api_client.delete_document("abc-123")
        assert result == payload
        mock_delete.assert_called_once()
        _, kwargs = mock_delete.call_args
        assert kwargs["params"]["document_id"] == "abc-123"

    def test_raises_on_not_found(self):
        mock = _mock_response({}, status_code=404)
        mock.raise_for_status.side_effect = Exception("404 Not Found")
        with patch("api_client.requests.delete", return_value=mock):
            with pytest.raises(Exception, match="404"):
                api_client.delete_document("no-such-id")
