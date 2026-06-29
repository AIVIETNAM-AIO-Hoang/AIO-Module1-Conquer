"""Streamlit RAG chat application."""

import requests
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
.source-score {
    float: right;
    color: #555555;
    font-size: 0.75rem;
    font-weight: normal;
    background: #eeeeee;
    border-radius: 10px;
    padding: 0 0.5rem;
}
.conn-pill {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: bold;
    border-radius: 12px;
    padding: 0.15rem 0.7rem;
    margin-bottom: 0.4rem;
}
.conn-ok { background: #e3f5e1; color: #1d7a2e; }
.conn-down { background: #fbe3e3; color: #b22020; }
</style>
"""


def _friendly_error(exc: Exception) -> str:
    """Translate an API exception into a clear, user-friendly message.

    Args:
        exc: The exception raised by an api_client call.

    Returns:
        A human-readable error string.
    """
    if isinstance(exc, requests.ConnectionError):
        return (
            "Cannot reach the API. Make sure the API server is running "
            "(uv run uvicorn app.main:app --reload)."
        )
    if isinstance(exc, requests.Timeout):
        return "The API took too long to respond. Please try again."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail")
        except ValueError:
            detail = None
        if exc.response.status_code == 404 and detail == "No relevant chunks found":
            return "No relevant content found. Have you uploaded any documents yet?"
        if detail:
            return f"API error: {detail}"
        return f"API error (status {exc.response.status_code})."
    return str(exc)


def _render_sources(sources: list[dict]) -> None:
    """Render source citations inside an expander, with retrieval score when available.

    Args:
        sources: List of source dicts with filename, content, and optional score.
    """
    with st.expander("Sources"):
        for src in sources:
            score = src.get("score")
            if isinstance(score, (int, float)):
                score_html = f'<span class="source-score">score {score:.2f}</span>'
            else:
                score_html = ""
            snippet = src["content"][:300]
            if len(src["content"]) > 300:
                snippet += "..."
            st.markdown(
                f'<div class="source-box">'
                f'<div class="source-filename">{src["filename"]}{score_html}</div>'
                f'<div class="source-label">{snippet}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _connection_status() -> bool:
    """Render an API connection indicator in the sidebar and return reachability.

    Returns:
        True if the API is reachable, False otherwise.
    """
    online = api_client.check_health()
    if online:
        st.sidebar.markdown(
            '<div class="conn-pill conn-ok">API: Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="conn-pill conn-down">API: Disconnected</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            "Start the API with: uv run uvicorn app.main:app --reload"
        )
    return online


def _sidebar(online: bool) -> None:
    """Render the document management sidebar.

    Args:
        online: Whether the API is currently reachable.
    """
    st.sidebar.title("Documents")

    st.sidebar.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=5,
        key="top_k",
        help="How many document chunks to use as context for each answer.",
    )

    st.sidebar.divider()

    uploaded = st.sidebar.file_uploader(
        "Upload a document", type=["pdf", "txt", "md"]
    )
    if st.sidebar.button("Upload", disabled=not online) and uploaded is not None:
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
                    st.error(_friendly_error(exc))

    st.sidebar.divider()
    st.sidebar.subheader("Uploaded documents")

    if not online:
        st.sidebar.info("Connect to the API to view your documents.")
        return

    try:
        docs = api_client.list_documents()
    except Exception as exc:
        st.sidebar.error(_friendly_error(exc))
        return

    if not docs:
        st.sidebar.caption("No documents yet. Upload one above to get started.")
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
                st.sidebar.error(_friendly_error(exc))


def _chat(online: bool) -> None:
    """Render the main chat interface.

    Args:
        online: Whether the API is currently reachable.
    """
    st.title("RAG Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    query = st.chat_input("Ask a question about your documents...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        if not online:
            msg = (
                "Cannot reach the API. Make sure the API server is running "
                "(uv run uvicorn app.main:app --reload)."
            )
            st.error(msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg, "sources": []}
            )
            return

        top_k = st.session_state.get("top_k", 5)
        with st.spinner("Thinking..."):
            try:
                result = api_client.prompt(query, top_k=top_k)
                answer = result["answer"]
                sources = result.get("sources", [])
                st.write(answer)
                if sources:
                    _render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                msg = _friendly_error(exc)
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "sources": []}
                )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="RAG Chat", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    online = _connection_status()
    _sidebar(online)
    _chat(online)


if __name__ == "__main__":
    main()
