from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.routers import documents, rag  # noqa: F401 - registers models with Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database on startup.

    Args:
        app: The FastAPI application instance.

    Returns:
        None
    """
    init_db()
    yield


app = FastAPI(title="ChatBot RAG API", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(rag.router)
