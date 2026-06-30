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


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases: empty query, top_k boundaries, no-document 404 (added)
# ──────────────────────────────────────────────────────────────────────────────


def test_retrieve_empty_query_is_accepted(client, uploaded_doc):
    """An empty query currently returns 200 (no input validation enforced).

    Pins down present behavior so a future decision to require a non-empty
    query becomes a deliberate, test-visible change.
    """
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/retrieve", json={"query": "", "top_k": 2})
    assert resp.status_code == 200


def test_retrieve_top_k_zero_returns_no_chunks(client, uploaded_doc):
    """top_k=0 retrieves zero chunks."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/retrieve", json={"query": "France", "top_k": 0})
    assert resp.status_code == 200
    assert resp.json()["chunks"] == []


def test_retrieve_large_top_k_returns_all_available(client, uploaded_doc):
    """Requesting more chunks than exist returns only what is available."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/retrieve", json={"query": "France", "top_k": 100})
    assert resp.status_code == 200
    # The fixture uploads exactly len(FAKE_CHUNKS) chunks.
    assert len(resp.json()["chunks"]) == len(FAKE_CHUNKS)


def test_retrieve_default_top_k_is_five(client, uploaded_doc):
    """Omitting top_k applies the schema default of 5 (capped by what exists)."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/retrieve", json={"query": "France"})
    assert resp.status_code == 200
    assert len(resp.json()["chunks"]) == len(FAKE_CHUNKS)  # < 5 available


def test_prompt_top_k_zero_returns_404(client, uploaded_doc):
    """top_k=0 yields no context, so /prompt reports 404 (no relevant chunks)."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/prompt", json={"query": "France", "top_k": 0})
    assert resp.status_code == 404


def test_prompt_with_no_documents_returns_404(client):
    """Prompting against an empty corpus returns 404, not an empty answer."""
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/prompt", json={"query": "anything", "top_k": 5})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No relevant chunks found"


def test_retrieve_with_no_documents_returns_empty_not_404(client):
    """/retrieve over an empty corpus returns 200 with an empty list.

    Contrasts with /prompt, which 404s — documenting the deliberate asymmetry
    between the two endpoints.
    """
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        resp = client.post("/api/rag/retrieve", json={"query": "anything", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["chunks"] == []


def test_prompt_sources_expose_retrieval_score(client, uploaded_doc):
    """Each source in a /prompt response carries its retrieval score.

    Regression guard for the duplicate-Source-schema bug: the score is computed
    in rag.py and must survive serialization through schemas.Source. If the
    duplicate (score-less) class is ever reintroduced, this test fails.
    """
    mock_llm = MagicMock()
    mock_llm.json.return_value = {"response": "Paris."}
    mock_llm.raise_for_status = MagicMock()

    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING), patch(
        _PATCH_LLM, return_value=mock_llm
    ):
        resp = client.post(
            "/api/rag/prompt",
            json={"query": "What is the capital of France?", "top_k": 2},
        )

    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert sources, "expected at least one source"
    for source in sources:
        assert "score" in source
        assert isinstance(source["score"], (int, float))
        assert 0.0 <= source["score"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end flow with value-level assertions (added)
# ──────────────────────────────────────────────────────────────────────────────


def test_e2e_full_flow_with_value_assertions(client):
    """upload -> list -> retrieve -> prompt -> remove, asserting values not just status.

    Strengthens the existing E2E (which mostly checks status codes and shapes)
    by verifying chunk counts, retrieved content, score range, that the LLM is
    actually grounded on the retrieved context, and that removal is complete
    across all read paths (list, retrieve, prompt).
    """
    # ── 1. Upload ──────────────────────────────────────────────────────────
    with patch(_PATCH_CHUNK, return_value=FAKE_CHUNKS), patch(
        _PATCH_EMBED_UPLOAD, return_value=FAKE_EMBEDDING
    ):
        up = client.post(
            "/api/documents/upload",
            files={"file": ("france.txt", b"content", "text/plain")},
        )
    assert up.status_code == 201
    up_body = up.json()
    doc_id = up_body["document_id"]
    assert up_body["filename"] == "france.txt"
    assert up_body["chunk_count"] == len(FAKE_CHUNKS)

    # ── 2. List ────────────────────────────────────────────────────────────
    listed = client.get("/api/documents/list").json()["documents"]
    assert len(listed) == 1
    entry = listed[0]
    assert entry["document_id"] == doc_id
    assert entry["chunk_count"] == len(FAKE_CHUNKS)
    assert entry["created_at"]  # timestamp present

    # ── 3. Retrieve ────────────────────────────────────────────────────────
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        ret = client.post(
            "/api/rag/retrieve",
            json={"query": "capital of France", "top_k": len(FAKE_CHUNKS)},
        )
    assert ret.status_code == 200
    chunks = ret.json()["chunks"]
    assert len(chunks) == len(FAKE_CHUNKS)
    assert {c["content"] for c in chunks} == set(FAKE_CHUNKS)
    for c in chunks:
        assert c["document_id"] == doc_id
        assert c["filename"] == "france.txt"
        assert 0.0 <= c["score"] <= 1.0
    # Results are ordered by descending similarity score.
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)

    # ── 4. Prompt (LLM grounded on retrieved context) ──────────────────────
    mock_llm = MagicMock()
    mock_llm.json.return_value = {"response": "Paris is the capital of France."}
    mock_llm.raise_for_status = MagicMock()
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING), patch(
        _PATCH_LLM, return_value=mock_llm
    ) as llm_post:
        prom = client.post(
            "/api/rag/prompt",
            json={"query": "What is the capital of France?", "top_k": len(FAKE_CHUNKS)},
        )
    assert prom.status_code == 200
    body = prom.json()
    assert body["answer"] == "Paris is the capital of France."
    assert {s["content"] for s in body["sources"]} == set(FAKE_CHUNKS)
    # The retrieved chunks must actually appear in the prompt sent to the LLM.
    sent_prompt = llm_post.call_args.kwargs["json"]["prompt"]
    for chunk in FAKE_CHUNKS:
        assert chunk in sent_prompt

    # ── 5. Remove ──────────────────────────────────────────────────────────
    rem = client.delete(f"/api/documents/remove?document_id={doc_id}")
    assert rem.status_code == 200
    assert rem.json()["deleted"] is True

    # ── 6. Confirm fully gone across every read path ───────────────────────
    assert client.get("/api/documents/list").json()["documents"] == []
    with patch(_PATCH_EMBED_SEARCH, return_value=FAKE_EMBEDDING):
        ret_after = client.post(
            "/api/rag/retrieve", json={"query": "France", "top_k": 5}
        )
        assert ret_after.json()["chunks"] == []
        prom_after = client.post(
            "/api/rag/prompt", json={"query": "France", "top_k": 5}
        )
        assert prom_after.status_code == 404
