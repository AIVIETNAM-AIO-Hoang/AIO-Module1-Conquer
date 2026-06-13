"""Streamlit front-end for the chatbotrag backend."""

import streamlit as st

import api_client

st.set_page_config(page_title="ChatBot RAG", layout="wide")

st.title("ChatBot RAG")


def _load_documents() -> list[dict]:
    """Fetch documents from the API, returning an empty list on error."""
    try:
        return api_client.list_documents()
    except Exception as exc:
        st.error(f"Could not reach API: {exc}")
        return []


# ── Upload section ────────────────────────────────────────────────────────────

st.header("Upload Document")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt", "md"],
    help="Supported formats: PDF, plain text, Markdown",
)

if st.button("Upload", disabled=uploaded_file is None):
    with st.spinner("Uploading and processing…"):
        try:
            result = api_client.upload_document(
                uploaded_file.read(), uploaded_file.name
            )
            st.success(
                f"Uploaded **{result['filename']}** — "
                f"{result['chunk_count']} chunk(s) created."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

st.divider()

# ── Document list section ─────────────────────────────────────────────────────

st.header("Documents")

documents = _load_documents()

if not documents:
    st.info("No documents uploaded yet.")
else:
    for doc in documents:
        col_name, col_chunks, col_date, col_action = st.columns([4, 1, 2, 1])
        col_name.write(doc["filename"])
        col_chunks.write(f"{doc['chunk_count']} chunks")
        col_date.write(doc["created_at"][:19].replace("T", " "))
        if col_action.button("Delete", key=doc["document_id"]):
            try:
                api_client.delete_document(doc["document_id"])
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")
