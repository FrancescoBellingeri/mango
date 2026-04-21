"""Shared fixtures for the Mango test suite."""

from __future__ import annotations

import os
from datetime import datetime

import mongomock
import pytest


# ---------------------------------------------------------------------------
# Marker registration and auto-skip
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and load .env so API keys are available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; fall back to OS environment only

    config.addinivalue_line("markers", "gemini: marks tests requiring Google API key")
    config.addinivalue_line("markers", "integration: marks end-to-end integration tests")


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001 — required by pytest hookspec
    items: list[pytest.Item],
) -> None:
    """Automatically skip provider tests when the required API key is absent.

    This mirrors Vanna's approach: mark the test at collection time so the
    skip reason appears in the summary without needing to run the test body.
    """
    for item in items:
        if "openai" in item.keywords and not os.getenv("OPENAI_API_KEY"):
            item.add_marker(
                pytest.mark.skip(reason="OPENAI_API_KEY environment variable not set")
            )
        if "anthropic" in item.keywords and not os.getenv("ANTHROPIC_API_KEY"):
            item.add_marker(
                pytest.mark.skip(reason="ANTHROPIC_API_KEY environment variable not set")
            )
        if "gemini" in item.keywords and not (
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set"
                )
            )
        if "mongodb" in item.keywords and not os.getenv("MONGODB_URI"):
            item.add_marker(
                pytest.mark.skip(reason="MONGODB_URI environment variable not set")
            )

from mango.integrations.mongodb import MongoRunner as MongoBackend
from mango.llm.base import LLMService
from mango.llm.models import LLMResponse, Message, ToolDef
from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService
from mango.tools.base import ToolRegistry
from mango.tools.mongo_tools import build_mongo_tools


# ---------------------------------------------------------------------------
# MongoDB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mongo_client():
    return mongomock.MongoClient()


@pytest.fixture
def mongo_db(mongo_client):
    """In-memory MongoDB with two test collections: users and orders."""
    db = mongo_client["testdb"]
    db["users"].insert_many([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ])
    db["orders"].insert_many([
        {"product": "Widget", "qty": 5, "user_id": "u1",
         "created_at": datetime(2024, 1, 15)},
        {"product": "Gadget", "qty": 2, "user_id": "u2",
         "created_at": datetime(2024, 3, 20)},
        {"product": "Widget", "qty": 10, "user_id": "u3",
         "created_at": datetime(2024, 6, 1)},
    ])
    return db


@pytest.fixture
def mongo_backend(mongo_client, mongo_db):
    """MongoBackend wired to the in-memory mongomock instance."""
    backend = MongoBackend()
    backend._client = mongo_client
    backend._db = mongo_db
    return backend


@pytest.fixture
def tool_registry(mongo_backend):
    """ToolRegistry populated with the standard MongoDB tools."""
    registry = ToolRegistry()
    for tool in build_mongo_tools(mongo_backend):
        registry.register(tool)
    return registry


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLMService(LLMService):
    """LLM that returns a pre-configured list of responses in order."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str = "",
    ) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system_prompt": system_prompt,
        })
        resp = self._responses[self._index]
        self._index += 1
        return resp

    def get_model_name(self) -> str:
        return "mock-model-v1"


@pytest.fixture
def MockLLM():
    """Return the MockLLMService class for use in tests."""
    return MockLLMService


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

_CHROMA_COLLECTION = "mango_memory"


@pytest.fixture
def memory_service():
    """ChromaMemoryService backed by an in-memory (ephemeral) ChromaDB.

    ChromaDB's EphemeralClient is a process-level singleton, so all tests
    share the same in-memory collection. We delete and recreate it before
    each test to guarantee a clean starting state.
    """
    svc = ChromaMemoryService(persist_dir=":memory:")
    # Wipe any data left by previous tests (singleton client).
    for col_name in (
        _CHROMA_COLLECTION,
        f"{_CHROMA_COLLECTION}_text",
        f"{_CHROMA_COLLECTION}_training",
    ):
        try:
            svc._client.delete_collection(col_name)
        except Exception:
            pass
    svc._collection = svc._client.get_or_create_collection(
        name=_CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    svc._text_collection = svc._client.get_or_create_collection(
        name=f"{_CHROMA_COLLECTION}_text",
        metadata={"hnsw:space": "cosine"},
    )
    svc._training_collection = svc._client.get_or_create_collection(
        name=f"{_CHROMA_COLLECTION}_training",
        metadata={"hnsw:space": "cosine"},
    )
    return svc
