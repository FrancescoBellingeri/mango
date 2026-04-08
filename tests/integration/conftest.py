"""
Integration test fixtures — real LLM services wired to a mongomock database.

All fixtures in this file require an active API key for the respective
provider. Tests are automatically skipped by the root conftest.py when
the key is absent (pytest_collection_modifyitems).

Models used are chosen to be fast and cost-efficient:
  - OpenAI   : MANGO_TEST_OPENAI_MODEL    (default: gpt-4o-mini)
  - Anthropic : MANGO_TEST_ANTHROPIC_MODEL (default: claude-haiku-4-5-20251001)
  - Gemini    : MANGO_TEST_GEMINI_MODEL    (default: gemini-2.0-flash)

Override any model via the corresponding environment variable.
"""

from __future__ import annotations

import os
from datetime import datetime

import mongomock
import pytest

from mango.agent.agent import MangoAgent
from mango.integrations.mongodb import MongoRunner as MongoBackend
from mango.tools.base import ToolRegistry
from mango.tools.mongo_tools import build_mongo_tools


# ---------------------------------------------------------------------------
# Shared test database (session-scoped — created once per pytest run)
# ---------------------------------------------------------------------------

#: Known facts about the test dataset used to assert LLM answers.
TEST_DB_FACTS = {
    "user_count": 3,
    "products": {"Widget", "Gadget"},
    "widget_total_qty": 15,   # 5 + 10
    "collections": {"users", "orders"},
}


@pytest.fixture(scope="session")
def integration_mongo_client():
    return mongomock.MongoClient()


@pytest.fixture(scope="session")
def integration_mongo_db(integration_mongo_client):
    """Session-scoped database. Populated once and reused by all integration tests."""
    db = integration_mongo_client["testdb"]
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


@pytest.fixture(scope="session")
def integration_backend(integration_mongo_client, integration_mongo_db):
    backend = MongoBackend()
    backend._client = integration_mongo_client
    backend._db = integration_mongo_db
    return backend


@pytest.fixture(scope="session")
def integration_registry(integration_backend):
    registry = ToolRegistry()
    for tool in build_mongo_tools(integration_backend):
        registry.register(tool)
    return registry


# ---------------------------------------------------------------------------
# LLM service fixtures — one per provider
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def openai_llm():
    """OpenAI LLM service (skipped if OPENAI_API_KEY is absent)."""
    from mango.llm.factory import build_llm
    model = os.getenv("MANGO_TEST_OPENAI_MODEL", "gpt-5.4-mini")
    return build_llm(
        provider="openai",
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@pytest.fixture(scope="session")
def anthropic_llm():
    """Anthropic LLM service (skipped if ANTHROPIC_API_KEY is absent)."""
    from mango.llm.factory import build_llm
    model = os.getenv("MANGO_TEST_ANTHROPIC_MODEL", "claude-haiku-4-5")
    return build_llm(
        provider="anthropic",
        model=model,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


@pytest.fixture(scope="session")
def gemini_llm():
    """Gemini LLM service (skipped if GOOGLE_API_KEY / GEMINI_API_KEY is absent)."""
    from mango.llm.factory import build_llm
    model = os.getenv("MANGO_TEST_GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    return build_llm(
        provider="gemini",
        model=model,
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Agent factory helper
# ---------------------------------------------------------------------------


def make_agent(llm, backend, registry) -> MangoAgent:
    """Build a MangoAgent without schema introspection (relies on tool calls)."""
    agent = MangoAgent(
        llm_service=llm,
        tool_registry=registry,
        db=backend,
        introspect=False,
    )
    agent.setup()
    return agent
