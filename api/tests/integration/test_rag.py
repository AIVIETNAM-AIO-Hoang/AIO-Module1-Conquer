from unittest.mock import MagicMock, patch

import pytest

FAKE_EMBEDDING = [0.1] * 768
FAKE_CHUNKS = ["The capital of France is Paris.", "France is in Western Europe."]

_PATCH_CHUNK = "app.routers.documents.chunk_document"
_PATCH_EMBED_UPLOAD = "app.routers.documents.get_embedding"
_PATCH_EMBED_SEARCH = "app.services.retrieval.get_embedding"
_PATCH_LLM = "app.services.llm.requests.post"


@pytest.fixture
def uploaded_doc(client):
    """Upload a document and return its document_id."""
    with patch(_PATCH_CHUNK, return_value=FAKE_CHUNKS), patch(
        _PATCH_EMBED_UPLOAD, return_value=FAKE_EMBEDDING
    ):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("france.txt", b"content", "text/plain")},
        )
    return resp.json()["document_id"]


def test_retrieve_returns_chunks(client, uploaded_doc):
    """Retrieve endpoint returns ranked chunks with expected shape."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post(
            "/api/rag/retrieve",
            json={"query": "What is the capital of France?", "top_k": 2},
        )

    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    assert len(chunks) > 0
    for chunk in chunks:
        assert "content" in chunk
        assert "document_id" in chunk
        assert "filename" in chunk
        assert "score" in chunk


def test_retrieve_respects_top_k(client, uploaded_doc):
    """Retrieve endpoint returns at most top_k results."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post(
            "/api/rag/retrieve",
            json={"query": "France", "top_k": 1},
        )
    assert resp.status_code == 200
    assert len(resp.json()["chunks"]) <= 1


def test_prompt_returns_answer_and_sources(client, uploaded_doc):
    """Prompt endpoint returns an answer string and source list."""
    mock_llm = MagicMock()
    mock_llm.json.return_value = {"response": "Paris is the capital of France."}
    mock_llm.raise_for_status = MagicMock()

    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING), patch(
        _PATCH_LLM, return_value=mock_llm
    ):
        resp = client.post(
            "/api/rag/prompt",
            json={"query": "What is the capital of France?", "top_k": 2},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert len(data["sources"]) > 0
    for source in data["sources"]:
        assert "document_id" in source
        assert "filename" in source
        assert "content" in source


def test_e2e_upload_list_retrieve_prompt_remove(client):
    """Full E2E flow: upload -> list -> retrieve -> prompt -> remove."""
    # Upload
    with patch(_PATCH_CHUNK, return_value=FAKE_CHUNKS), patch(
        _PATCH_EMBED_UPLOAD, return_value=FAKE_EMBEDDING
    ):
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("e2e.txt", b"content", "text/plain")},
        )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["document_id"]

    # List
    list_resp = client.get("/api/documents/list")
    assert any(d["document_id"] == doc_id for d in list_resp.json()["documents"])

    # Retrieve
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        ret_resp = client.post("/api/rag/retrieve", json={"query": "France", "top_k": 2})
    assert ret_resp.status_code == 200
    assert len(ret_resp.json()["chunks"]) > 0

    # Prompt
    mock_llm = MagicMock()
    mock_llm.json.return_value = {"response": "An answer."}
    mock_llm.raise_for_status = MagicMock()
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING), patch(
        _PATCH_LLM, return_value=mock_llm
    ):
        prom_resp = client.post("/api/rag/prompt", json={"query": "France", "top_k": 2})
    assert prom_resp.status_code == 200

    # Remove
    del_resp = client.delete(f"/api/documents/remove?document_id={doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Confirm gone
    final_list = client.get("/api/documents/list")
    assert not any(d["document_id"] == doc_id for d in final_list.json()["documents"])
