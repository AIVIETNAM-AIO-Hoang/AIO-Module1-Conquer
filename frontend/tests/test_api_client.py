"""Unit tests for api_client, all HTTP calls mocked."""

import pytest
import api_client


def _make_response(mocker, json_data: dict, status_code: int = 200):
    """Build a mock requests.Response."""
    mock = mocker.MagicMock()
    mock.json.return_value = json_data
    mock.status_code = status_code
    mock.raise_for_status.return_value = None
    return mock


class TestUploadDocument:
    def test_returns_upload_response(self, mocker):
        payload = {"document_id": "abc-123", "filename": "doc.txt", "chunk_count": 4}
        mock_post = mocker.patch("api_client.requests.post", return_value=_make_response(mocker, payload))

        result = api_client.upload_document(b"hello", "doc.txt", "text/plain")

        assert result == payload
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "/api/documents/upload" in call_kwargs[0][0]
        assert "file" in call_kwargs[1]["files"]

    def test_raises_on_http_error(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("422 Unprocessable")
        mocker.patch("api_client.requests.post", return_value=mock_resp)

        with pytest.raises(Exception, match="422"):
            api_client.upload_document(b"", "bad.xyz", "application/octet-stream")


class TestListDocuments:
    def test_returns_documents_list(self, mocker):
        docs = [
            {"document_id": "id-1", "filename": "a.txt", "chunk_count": 2, "created_at": "2024-01-01T00:00:00"},
        ]
        mocker.patch("api_client.requests.get", return_value=_make_response(mocker, {"documents": docs}))

        result = api_client.list_documents()

        assert result == docs

    def test_raises_on_http_error(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500")
        mocker.patch("api_client.requests.get", return_value=mock_resp)

        with pytest.raises(Exception, match="500"):
            api_client.list_documents()


class TestRemoveDocument:
    def test_returns_delete_response(self, mocker):
        payload = {"document_id": "id-1", "deleted": True}
        mock_delete = mocker.patch(
            "api_client.requests.delete", return_value=_make_response(mocker, payload)
        )

        result = api_client.remove_document("id-1")

        assert result == payload
        call_kwargs = mock_delete.call_args
        assert call_kwargs[1]["params"] == {"document_id": "id-1"}

    def test_raises_on_http_error(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        mocker.patch("api_client.requests.delete", return_value=mock_resp)

        with pytest.raises(Exception, match="404"):
            api_client.remove_document("missing-id")


class TestPrompt:
    def test_returns_prompt_response(self, mocker):
        payload = {
            "answer": "The sky is blue.",
            "sources": [{"document_id": "id-1", "filename": "sky.txt", "content": "..."}],
        }
        mock_post = mocker.patch("api_client.requests.post", return_value=_make_response(mocker, payload))

        result = api_client.prompt("Why is the sky blue?", top_k=3)

        assert result["answer"] == "The sky is blue."
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"] == {"query": "Why is the sky blue?", "top_k": 3}

    def test_default_top_k(self, mocker):
        payload = {"answer": "answer", "sources": []}
        mock_post = mocker.patch("api_client.requests.post", return_value=_make_response(mocker, payload))

        api_client.prompt("hello")

        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["top_k"] == 5

    def test_raises_on_http_error(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 No chunks")
        mocker.patch("api_client.requests.post", return_value=mock_resp)

        with pytest.raises(Exception, match="404"):
            api_client.prompt("nothing here")
