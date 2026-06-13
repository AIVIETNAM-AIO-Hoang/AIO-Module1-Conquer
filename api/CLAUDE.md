# Chat Bot with RAG - Project Spec

A single-user FastAPI microservice that ingests documents, stores them with
embeddings in PostgreSQL (pgvector), and answers prompts with grounded,
source-attributed responses using a local Ollama LLM.

This stack is deliberately kept beginner-friendly: everything is synchronous,
and all AI models (chat and embeddings) run in one place, Ollama. The mental
model is four steps: FastAPI takes the request, Ollama makes the vectors,
Postgres finds the closest chunks, Ollama writes the answer.

## Tech stack

- Language: Python 3.12+
- Package manager: uv (pin exact versions in pyproject.toml)
- Web framework: FastAPI, synchronous endpoints (plain def, no async/await)
- Orchestration: LangChain (latest 0.3+ line), used only for semantic chunking
- Database: PostgreSQL with the pgvector extension
- DB access: SQLAlchemy 2.x (synchronous) with psycopg
- Config: pydantic-settings, loaded from .env
- Embedding model: nomic-embed-text, served by Ollama
- Chunking: LangChain SemanticChunker (uses the Ollama embedding model)
- Retrieval: dense similarity search over pgvector
- LLM: local TinyLlama

## Local prerequisites

The developer is expected to have, running locally:

- PostgreSQL with the pgvector extension installed and enabled.
- TinyLlama running, with the nomic-embed-text embedding
  model pulled.

Document these exact setup commands in the README.

## Data model

Two tables.

- documents: id (uuid, pk), filename (text), content_type (text),
  created_at (timestamptz)
- chunks: id (uuid, pk), document_id (uuid, fk -> documents.id, on delete
  cascade), content (text), embedding (vector), chunk_index (int)

Create an HNSW index on chunks.embedding for cosine distance.

## API contract

All endpoints are prefixed with /api. Request and response bodies are Pydantic
models. Endpoints are synchronous (def). Use appropriate HTTP status codes; do
not add auth.

1. POST /api/documents/upload
   - Accepts a multipart file upload. Supported types: .pdf, .txt, .md.
   - Parses, semantically chunks, embeds each chunk, stores document + chunks.
   - Returns: document_id, filename, chunk_count.

2. GET /api/documents/list
   - Returns a list of {document_id, filename, chunk_count, created_at}.

3. DELETE /api/documents/remove
   - Accepts a document_id.
   - Deletes the document and its chunks (cascade).
   - Returns: {document_id, deleted: true}.

4. POST /api/rag/retrieve
   - Accepts {query, top_k}.
   - Dense search over pgvector. No LLM call.
   - Returns ranked chunks with {content, document_id, filename, score}.

5. POST /api/rag/prompt
   - Accepts {query, top_k}.
   - Runs retrieval (as above), then generates a grounded answer with the
     local Ollama model.
   - Returns {answer, sources: [{document_id, filename, content}]}.
     The answer must be grounded only in the retrieved chunks; cite sources.

## Configuration

All configuration via pydantic-settings from .env. Provide a committed
.env.example. Keys at minimum: DATABASE_URL, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL,
OLLAMA_EMBEDDING_MODEL. Add .env to .gitignore.

## Coding standards

1. Use latest mutually compatible library versions, pinned, with idiomatic
   modern Python. Keep everything synchronous.
2. Apply SOLID, but keep it simple. NEVER over-engineer, ALWAYS simplify, NO
   unnecessary defensive programming. Validate genuine contract violations
   (e.g. missing file, unknown document_id) and let everything else fail
   naturally. No extra features beyond this spec.
3. Be concise. Keep the README minimal. No emojis, ever.
4. Every class and function has a docstring with Args, Returns, and Raises (when
   it raises), following this form:

   def divide(dividend: float, divisor: float) -> float:
   """Divide a number by another number.
   Args:
   dividend: The number to be divided.
   divisor: The number to divide by.
   Returns:
   The resulting quotient.
   Raises:
   ValueError: If the divisor is zero.
   """
   if divisor == 0:
   raise ValueError("Cannot divide by zero.")
   return dividend / divisor

## Testing strategy

- Unit tests with pytest. Mock the Ollama embedding and chat calls so unit
  tests are fast and offline.
- Integration tests with pytest + FastAPI TestClient against a real test
  PostgreSQL database with pgvector.
- Do NOT use Playwright; there is no browser UI in this service.

## Out of scope (non-goals)

Do not build any of the following: authentication or authorization, rate
limiting, multi-tenancy or per-user isolation, streaming responses, async
endpoints, a frontend, background job queues, caching layers, or hybrid/sparse
retrieval. Single user, synchronous request/response only.

## Optional phase 2 (only after the MVP is complete and tested)

Add a cross-encoder reranker to improve retrieval precision. This is the one
enhancement that reintroduces a torch / sentence-transformers dependency, so it
is kept out of the MVP on purpose.

- Model: bge-reranker-base (cross-encoder).
- Encapsulate it entirely in a single rerank(query, chunks) function so the
  added complexity lives in one file.
- Call it in /api/rag/retrieve and /api/rag/prompt after the dense search, to
  reorder the top candidates before they are returned or sent to the LLM.

Do not start phase 2 until every MVP acceptance criterion below is met.

## Delivery strategy

1. Write a phased plan with check-off success criteria per phase, covering:
   project scaffolding (pyproject.toml, .gitignore, .env.example, package
   layout), database schema and init, each endpoint, and rigorous unit tests.
2. Execute the plan, ensuring every criterion is met.
3. Carry out integration testing with pytest + TestClient, fixing all defects.
4. Define E2E tests and their success criteria (full flow: upload -> list ->
   retrieve -> prompt -> remove).
5. Only complete when the MVP is finished and tested, with the server running
   and ready for the user.

## Acceptance criteria (MVP done when all true)

- All five endpoints implemented and returning the contracted shapes.
- Upload chunks semantically, embeds via Ollama, and persists to
  PostgreSQL/pgvector.
- /api/rag/prompt returns an answer grounded only in retrieved chunks, with
  source attribution.
- Unit and integration tests pass.
- Server starts cleanly with documented commands; README is minimal and
  emoji-free.
