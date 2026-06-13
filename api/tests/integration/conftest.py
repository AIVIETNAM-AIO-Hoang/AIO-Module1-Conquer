import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base, get_session
from app.main import app

# Derive test DB URL from the configured one, allow explicit override.
_base_url = settings.DATABASE_URL.rsplit("/", 1)[0]
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", f"{_base_url}/chatbotrag_test")

test_engine = create_engine(TEST_DATABASE_URL)
TestSessionLocal = sessionmaker(bind=test_engine)


def override_get_session():
    """Yield a session pointed at the test database."""
    with TestSessionLocal() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create extension, tables, and HNSW index once for the test session."""
    import app.models  # noqa: F401 - registers models with Base

    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(test_engine)
    with test_engine.connect() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
        conn.commit()
    yield
    Base.metadata.drop_all(test_engine)


@pytest.fixture(autouse=True)
def clean_db():
    """Truncate all rows between tests."""
    yield
    with TestSessionLocal() as session:
        from app.models import Document

        session.query(Document).delete()
        session.commit()


@pytest.fixture
def client():
    """Return a TestClient wired to the test database."""
    app.dependency_overrides[get_session] = override_get_session
    with patch("app.main.init_db"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
