import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
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
    """Create extension, tables, and HNSW index once for the test session.

    If the test database is not reachable, the whole integration suite is
    skipped (not errored) with an actionable message, so running the full repo
    without infrastructure stays clean. To actually exercise these tests, start
    PostgreSQL with the pgvector extension and create the test database, e.g.:

        docker run -d --name pgvector-test -p 5432:5432 \\
            -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test \\
            -e POSTGRES_DB=test pgvector/pgvector:pg16
        docker exec pgvector-test psql -U test -d test \\
            -c "CREATE DATABASE chatbotrag_test;"

    (Match the credentials/host to your DATABASE_URL, or set TEST_DATABASE_URL.)
    """
    import app.models  # noqa: F401 - registers models with Base

    try:
        with test_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except OperationalError as exc:
        pytest.skip(
            "Integration database not reachable at "
            f"{TEST_DATABASE_URL!r}. Start PostgreSQL+pgvector and create the "
            "test database, or set TEST_DATABASE_URL. "
            f"({exc.orig.__class__.__name__})",
            allow_module_level=False,
        )
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
def clean_db(setup_test_db):
    """Truncate all rows between tests.

    Depends on setup_test_db so that, when the database is unreachable and the
    suite is skipped, this teardown does not attempt its own connection.
    """
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
