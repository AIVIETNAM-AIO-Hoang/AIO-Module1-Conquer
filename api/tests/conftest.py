import os

# Provide defaults so unit tests can import app modules without a real DB or Ollama.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_LLM_MODEL", "minimax-m3:cloud")
os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
