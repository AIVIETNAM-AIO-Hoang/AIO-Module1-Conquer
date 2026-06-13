# API Build Plan

## Phase 1: Project Scaffolding

### Tasks
- [x] Create `pyproject.toml` with pinned dependencies (fastapi, uvicorn, sqlalchemy, psycopg, langchain, langchain-community, langchain-ollama, langchain-experimental, pgvector, pydantic-settings, python-multipart, pypdf)
- [x] Create `.env.example` with DATABASE_URL, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBEDDING_MODEL
- [x] Create `.gitignore` covering `.env`, `__pycache__`, `.venv`, `*.pyc`, `.pytest_cache`
- [x] Establish package layout:
  ```
  api/
    app/
      __init__.py
      main.py          # FastAPI app creation and router registration
      config.py        # pydantic-settings Settings class
      database.py      # SQLAlchemy engine, session factory
      models.py        # ORM models: Document, Chunk
      schemas.py       # Pydantic request/response models
      routers/
        __init__.py
        documents.py   # upload, list, delete endpoints
        rag.py         # retrieve, prompt endpoints
      services/
        __init__.py
        embedding.py   # Ollama embedding calls
        chunking.py    # LangChain SemanticChunker wrapper
        retrieval.py   # pgvector similarity search
        llm.py         # Ollama LLM call
    tests/
      __init__.py
      conftest.py
      unit/
        __init__.py
        test_embedding.py
        test_chunking.py
        test_retrieval.py
        test_llm.py
      integration/
        __init__.py
        conftest.py
        test_documents.py
        test_rag.py
  ```

### Success Criteria
- [x] `uv sync` installs all dependencies without errors
- [x] `.env` is gitignored; `.env.example` is committed

---

## Phase 2: Database Schema and Initialization

### Tasks
- [x] Implement `app/models.py`: Document and Chunk ORM models with UUID PKs, FK with cascade delete
- [x] Implement `app/database.py`: synchronous SQLAlchemy engine from DATABASE_URL, session factory, `init_db()` that creates tables and HNSW index on `chunks.embedding`
- [x] Call `init_db()` on app startup via `lifespan` event
- [x] Confirm pgvector extension is enabled before table creation

### Success Criteria
- [ ] Running `init_db()` against a live PostgreSQL instance creates both tables
- [ ] HNSW index exists on `chunks.embedding` for cosine distance
- [ ] Cascade delete: removing a Document row also removes its Chunk rows

---

## Phase 3: Core Services

### Tasks
- [x] `app/services/embedding.py`: `get_embedding(text: str) -> list[float]`
- [x] `app/services/chunking.py`: `chunk_document(text: str) -> list[str]`
- [x] `app/services/retrieval.py`: `search_chunks(query, top_k, session) -> list[dict]`
- [x] `app/services/llm.py`: `generate_answer(query, context_chunks) -> str`

### Success Criteria
- [x] Services implemented; validated by unit tests

---

## Phase 4: Document Endpoints

### Tasks
- [x] `POST /api/documents/upload`
- [x] `GET /api/documents/list`
- [x] `DELETE /api/documents/remove`
- [x] Validation: unsupported file types return 422; unknown document_id returns 404

---

## Phase 5: RAG Endpoints

### Tasks
- [x] `POST /api/rag/retrieve`
- [x] `POST /api/rag/prompt`
- [x] Grounded system prompt with source citation

---

## Phase 6: Unit Tests

### Tasks
- [x] `test_embedding.py`
- [x] `test_chunking.py`
- [x] `test_retrieval.py`
- [x] `test_llm.py`

### Success Criteria
- [x] All 9 unit tests pass offline (`pytest tests/unit` exits 0)

---

## Phase 7: Integration Tests *(skipped by user)*

Tests written and ready in `tests/integration/`. Run when PostgreSQL test DB is available:

```
psql -U postgres -c "CREATE DATABASE chatbotrag_test;"
uv run pytest tests/integration -v
```

---

## Phase 8: Final Checks

### Tasks
- [x] Write minimal `README.md`
- [ ] Confirm server starts cleanly with `uvicorn app.main:app --reload`
- [ ] Verify `/docs` loads

### To start the server
1. Copy `.env.example` to `.env` and fill in your values
2. `uv run uvicorn app.main:app --reload`
