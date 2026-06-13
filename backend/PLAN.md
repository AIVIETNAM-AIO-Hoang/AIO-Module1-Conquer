# Backend Build Plan

## Phase 1: Project Scaffolding

### Tasks
- [x] Create `pyproject.toml` with pinned deps: streamlit, requests, python-dotenv
- [x] Create `.env.example` with `API_BASE_URL=http://localhost:8000`
- [x] Create `.gitignore` covering `.env`, `__pycache__`, `.venv`, `*.pyc`, `.pytest_cache`
- [x] Package layout: `app.py`, `api_client.py`, `tests/`

### Success Criteria
- [x] `uv sync` installs without errors
- [x] `.env` is gitignored

---

## Phase 2: API Client

### Tasks
- [x] `api_client.py`: `upload_document(file_bytes, filename) -> dict`
- [x] `api_client.py`: `list_documents() -> list[dict]`
- [x] `api_client.py`: `delete_document(document_id) -> dict`

### Success Criteria
- [x] Each function calls the correct endpoint with the correct method/payload
- [x] HTTP errors surface as exceptions

---

## Phase 3: Streamlit UI

### Tasks
- [x] Upload section: file uploader (pdf/txt/md), upload button, success message
- [x] Document list section: filename, chunk count, created_at, Delete button per row

### Success Criteria
- [x] Document list renders on load
- [x] Upload triggers POST /api/documents/upload; list refreshes after
- [x] Delete triggers DELETE /api/documents/remove; list refreshes after
- [x] Unsupported file types blocked at the uploader level

---

## Phase 4: Unit Tests

### Tasks
- [x] `tests/test_api_client.py`: 6 tests covering all three functions (success + error)

### Success Criteria
- [x] All 6 tests pass offline (`pytest tests/` exits 0)

---

## To run the backend

1. Copy `.env.example` to `.env` and set `API_BASE_URL` to the running API address.
2. `uv run streamlit run app.py`
