from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""


engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def get_session():
    """Yield a database session for use as a FastAPI dependency.

    Returns:
        A SQLAlchemy Session instance.
    """
    with SessionLocal() as session:
        yield session


def init_db() -> None:
    """Create all tables and required extensions/indexes on first run.

    Returns:
        None

    Raises:
        sqlalchemy.exc.OperationalError: If the database is unreachable.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
        conn.commit()
