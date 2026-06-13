"""Streamlit RAG chat application."""

import streamlit as st

import api_client

_CONTENT_TYPE_MAP = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "md": "text/markdown",
}

_CSS = """
<style>
h1, h2, h3 { color: #032147; }
.stButton > button {
    background-color: #753991;
    color: white;
    border: none;
    border-radius: 4px;
}
.stButton > button:hover { background-color: #5e2e75; }
.source-box {
    border-left: 3px solid #ecad0a;
    padding: 0.4rem 0.8rem;
    margin-bottom: 0.5rem;
    background: #f9f9f9;
}
.source-label { color: #888888; font-size: 0.8rem; }
.source-filename { color: #209dd7; font-weight: bold; }
</style>
"""


def _render_sources(sources: list[dict]) -> None:
    """Render source citations inside an expander.

    Args:
        sources: List of source dicts with filename and content.
    """
    with st.expander("Sources"):
        for src in sources:
            st.markdown(
                f'<div class="source-box">'
                f'<div class="source-filename">{src["filename"]}</div>'
                f'<div class="source-label">{src["content"][:300]}{"..." if len(src["content"]) > 300 else ""}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _sidebar() -> None:
    """Render the document management sidebar."""
    st.sidebar.title("Documents")

    uploaded = st.sidebar.file_uploader(
        "Upload a document", type=["pdf", "txt", "md"]
    )
    if st.sidebar.button("Upload") and uploaded is not None:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        content_type = _CONTENT_TYPE_MAP.get(ext, "text/plain")
        with st.sidebar:
            with st.spinner("Uploading..."):
                try:
                    result = api_client.upload_document(
                        uploaded.read(), uploaded.name, content_type
                    )
                    st.success(f"Uploaded: {result['chunk_count']} chunks")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.sidebar.divider()
    st.sidebar.subheader("Uploaded documents")

    try:
        docs = api_client.list_documents()
    except Exception as exc:
        st.sidebar.error(str(exc))
        return

    if not docs:
        st.sidebar.caption("No documents yet.")
        return

    for doc in docs:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.markdown(
            f"**{doc['filename']}**  \n"
            f"<span style='color:#888888;font-size:0.75rem'>{doc['chunk_count']} chunks</span>",
            unsafe_allow_html=True,
        )
        if col2.button("Del", key=f"del_{doc['document_id']}"):
            try:
                api_client.remove_document(doc["document_id"])
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))


def _chat() -> None:
    """Render the main chat interface."""
    st.title("RAG Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = api_client.prompt(query)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    st.write(answer)
                    if sources:
                        _render_sources(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    err = str(exc)
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err, "sources": []}
                    )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="RAG Chat", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    _sidebar()
    _chat()


if __name__ == "__main__":
    main()
