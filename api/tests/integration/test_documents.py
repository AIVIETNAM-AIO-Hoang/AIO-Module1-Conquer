import uuid
from unittest.mock import patch

FAKE_EMBEDDING = [0.1] * 768
FAKE_CHUNKS = ["First semantic chunk.", "Second semantic chunk."]

_PATCH_CHUNK = "app.routers.documents.chunk_document"
_PATCH_EMBED = "app.routers.documents.get_embedding"


def _upload(client, filename="test.txt", content=b"Some content.", content_type="text/plain"):
    with patch(_PATCH_CHUNK, return_value=FAKE_CHUNKS), patch(
        _PATCH_EMBED, return_value=FAKE_EMBEDDING
    ):
        return client.post(
            "/api/documents/upload",
            files={"file": (filename, content, content_type)},
        )


def test_upload_txt_returns_201(client):
    """Uploading a .txt file creates a document and returns 201."""
    resp = _upload(client)
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "test.txt"
    assert data["chunk_count"] == 2
    assert "document_id" in data


def test_upload_pdf_returns_201(client):
    """Uploading a .pdf file is accepted."""
    from pypdf import PdfWriter
    import io

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    with patch(_PATCH_CHUNK, return_value=["pdf chunk"]), patch(
        _PATCH_EMBED, return_value=FAKE_EMBEDDING
    ):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
    assert resp.status_code == 201
    assert resp.json()["filename"] == "doc.pdf"


def test_upload_unsupported_type_returns_422(client):
    """Uploading an unsupported file type returns 422."""
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("binary.exe", b"data", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_list_returns_uploaded_documents(client):
    """List endpoint includes the uploaded document with correct chunk count."""
    _upload(client)
    resp = client.get("/api/documents/list")
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "test.txt"
    assert docs[0]["chunk_count"] == 2


def test_list_empty_when_no_documents(client):
    """List endpoint returns an empty list when no documents exist."""
    resp = client.get("/api/documents/list")
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


def test_delete_removes_document(client):
    """Delete endpoint removes the document; subsequent list is empty."""
    doc_id = _upload(client).json()["document_id"]
    resp = client.delete(f"/api/documents/remove?document_id={doc_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    list_resp = client.get("/api/documents/list")
    assert list_resp.json()["documents"] == []


def test_delete_unknown_returns_404(client):
    """Deleting a non-existent document returns 404."""
    resp = client.delete(f"/api/documents/remove?document_id={uuid.uuid4()}")
    assert resp.status_code == 404
