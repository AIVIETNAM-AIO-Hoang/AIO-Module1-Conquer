# ChatBot RAG API

A synchronous FastAPI microservice for document ingestion and retrieval-augmented generation using PostgreSQL/pgvector and Ollama.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv): `pip install uv`
- PostgreSQL with the pgvector extension:
  ```sql
  CREATE EXTENSION vector;
  ```
- Ollama running locally with models pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull tinyllama
  ```

## Setup

```bash
uv sync --group dev
cp .env.example .env
# Edit .env with your DATABASE_URL and Ollama settings
```

Tables and indexes are created automatically on first startup.

## Run

```bash
uv run uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## Tests

```bash
# Unit tests (no external services required)
uv run pytest tests/unit

# Integration tests (requires PostgreSQL with pgvector and the chatbotrag_test database)
uv run pytest tests/integration
```

Create the test database before running integration tests:

```sql
CREATE DATABASE chatbotrag_test;
```
