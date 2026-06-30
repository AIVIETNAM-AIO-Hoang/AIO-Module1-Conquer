"""Unit tests for the pure helpers in app.routers.documents.

These exercise content-type resolution and text extraction directly, with no
database, Ollama, or HTTP involved — so they are fast and fully hermetic.
They cover edge cases the endpoint tests only touch indirectly: empty files,
unsupported formats, and Unicode / invalid-byte handling.
"""

import io

import pytest
from fastapi import HTTPException

from app.routers.documents import _extract_text, _resolve_content_type


# ──────────────────────────────────────────────────────────────────────────────
# _resolve_content_type — extension first, declared type as fallback
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("notes.txt", "text/plain"),
        ("README.md", "text/markdown"),
        ("paper.pdf", "application/pdf"),
        ("UPPER.PDF", "application/pdf"),          # case-insensitive
        ("report.pdf.txt", "text/plain"),          # last extension wins
    ],
)
def test_resolve_content_type_from_extension(filename, expected):
    """Extension drives the resolved content type, regardless of declared type."""
    assert _resolve_content_type(filename, declared=None) == expected


def test_resolve_content_type_falls_back_to_declared_when_no_extension():
    """With no usable extension, a supported declared MIME type is accepted."""
    assert _resolve_content_type("datafile", "application/pdf") == "application/pdf"
    assert _resolve_content_type(None, "text/markdown") == "text/markdown"


def test_resolve_content_type_extension_beats_declared():
    """When both are present, the extension takes precedence over declared type."""
    # Declared says PDF, but the .md extension should win.
    assert _resolve_content_type("guide.md", "application/pdf") == "text/markdown"


@pytest.mark.parametrize(
    "filename, declared",
    [
        ("malware.exe", "application/octet-stream"),
        ("archive.zip", "application/zip"),
        ("photo.png", "image/png"),
        (None, "application/octet-stream"),
        ("no_extension", None),
    ],
)
def test_resolve_content_type_unsupported_raises_422(filename, declared):
    """Unsupported extension and declared type both fail with HTTP 422."""
    with pytest.raises(HTTPException) as exc_info:
        _resolve_content_type(filename, declared)
    assert exc_info.value.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# _extract_text — plain text path (empty + Unicode)
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_text_empty_plain_text_returns_empty_string():
    """An empty text file extracts to an empty string, not an error."""
    assert _extract_text(b"", "text/plain") == ""


def test_extract_text_preserves_unicode():
    """UTF-8 content (Vietnamese, CJK, emoji) round-trips intact."""
    original = "Xin chào, thế giới 世界 🌍 — café naïve"
    extracted = _extract_text(original.encode("utf-8"), "text/plain")
    assert extracted == original


def test_extract_text_replaces_invalid_bytes_without_raising():
    """Invalid UTF-8 bytes are replaced (errors='replace'), never crash."""
    # 0xff / 0xfe are not valid UTF-8 lead bytes.
    extracted = _extract_text(b"\xff\xfe valid tail", "text/plain")
    assert "valid tail" in extracted
    assert "\ufffd" in extracted  # the Unicode replacement character


def test_extract_text_markdown_is_decoded_as_text():
    """Markdown content is decoded as UTF-8 text like plain text."""
    md = "# Tiêu đề\n\n- mục 1\n- mục 2"
    assert _extract_text(md.encode("utf-8"), "text/markdown") == md


# ──────────────────────────────────────────────────────────────────────────────
# _extract_text — PDF path (empty / textless)
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_text_blank_pdf_returns_empty_string():
    """A structurally valid PDF with no text yields an empty string."""
    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buf)

    assert _extract_text(buf.getvalue(), "application/pdf") == ""


def test_extract_text_empty_pdf_bytes_raises():
    """Zero-byte 'PDF' content is not a valid PDF and surfaces as an error.

    Documents current behavior: the upload endpoint does not pre-validate PDF
    structure, so empty bytes propagate a pypdf read error rather than a clean
    422. Worth surfacing as a product decision.
    """
    with pytest.raises(Exception):
        _extract_text(b"", "application/pdf")
