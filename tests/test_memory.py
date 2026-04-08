"""Tests for mango.memory.chroma — ChromaMemoryService (ephemeral)."""

from __future__ import annotations

from mango.memory.models import MemoryEntry, TextMemoryEntry
from mango.integrations.chromadb import make_entry_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    question: str = "How many users are there?",
    tool_name: str = "run_mql",
    tool_args: dict | None = None,
    result_summary: str = "There are 42 users.",
) -> MemoryEntry:
    return MemoryEntry(
        id=make_entry_id(),
        question=question,
        tool_name=tool_name,
        tool_args=tool_args or {"operation": "count", "collection": "users"},
        result_summary=result_summary,
    )


# ---------------------------------------------------------------------------
# Tool-usage memory
# ---------------------------------------------------------------------------


class TestChromaMemoryService:
    async def test_initial_count_is_zero(self, memory_service):
        assert memory_service.count() == 0

    async def test_store_increases_count(self, memory_service):
        await memory_service.store(_make_entry())
        assert memory_service.count() == 1

    async def test_store_multiple(self, memory_service):
        for i in range(3):
            await memory_service.store(_make_entry(question=f"Question {i}"))
        assert memory_service.count() == 3

    async def test_retrieve_from_empty_returns_empty(self, memory_service):
        results = await memory_service.retrieve("how many users?")
        assert results == []

    async def test_retrieve_similar_entry(self, memory_service):
        entry = _make_entry(question="How many users are registered?")
        await memory_service.store(entry)
        results = await memory_service.retrieve(
            "Count the number of registered users", similarity_threshold=0.0
        )
        assert len(results) >= 1
        assert results[0].tool_name == "run_mql"

    async def test_retrieve_result_has_similarity_score(self, memory_service):
        await memory_service.store(_make_entry())
        results = await memory_service.retrieve("How many users?", similarity_threshold=0.0)
        assert len(results) >= 1
        assert 0.0 <= results[0].similarity <= 1.0

    async def test_retrieve_respects_top_k(self, memory_service):
        for i in range(5):
            await memory_service.store(_make_entry(question=f"Query about users number {i}"))
        results = await memory_service.retrieve(
            "users query", top_k=2, similarity_threshold=0.0
        )
        assert len(results) <= 2

    async def test_retrieve_filters_by_similarity_threshold(self, memory_service):
        await memory_service.store(_make_entry(question="What is the weather in Rome?"))
        results = await memory_service.retrieve(
            "aggregate MongoDB pipeline group by date",
            similarity_threshold=0.99,
        )
        assert results == []

    async def test_upsert_same_id_does_not_duplicate(self, memory_service):
        entry = _make_entry()
        await memory_service.store(entry)
        await memory_service.store(entry)  # same ID
        assert memory_service.count() == 1

    async def test_delete_removes_entry(self, memory_service):
        entry = _make_entry()
        await memory_service.store(entry)
        assert memory_service.count() == 1
        await memory_service.delete(entry.id)
        assert memory_service.count() == 0

    async def test_delete_nonexistent_is_noop(self, memory_service):
        await memory_service.delete("nonexistent-id-xyz")
        assert memory_service.count() == 0

    async def test_stored_tool_args_round_trip(self, memory_service):
        args = {"operation": "find", "collection": "orders", "filter": {"status": "open"}}
        entry = _make_entry(tool_args=args)
        await memory_service.store(entry)
        results = await memory_service.retrieve(
            "find open orders", top_k=1, similarity_threshold=0.0
        )
        assert len(results) == 1
        assert results[0].tool_args == args


# ---------------------------------------------------------------------------
# Text memory
# ---------------------------------------------------------------------------


class TestTextMemory:
    async def test_save_text_returns_id(self, memory_service):
        entry_id = await memory_service.save_text("revenue means total_amount field")
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    async def test_save_text_does_not_affect_tool_count(self, memory_service):
        await memory_service.save_text("some glossary entry")
        assert memory_service.count() == 0

    async def test_search_text_empty_returns_empty(self, memory_service):
        results = await memory_service.search_text("revenue")
        assert results == []

    async def test_search_text_finds_saved_entry(self, memory_service):
        await memory_service.save_text("'active customer' means ordered in the last 90 days")
        results = await memory_service.search_text(
            "what is an active customer?", similarity_threshold=0.0
        )
        assert len(results) >= 1
        assert isinstance(results[0], TextMemoryEntry)
        assert "active customer" in results[0].text

    async def test_search_text_has_similarity_score(self, memory_service):
        await memory_service.save_text("revenue is the total_amount field in orders")
        results = await memory_service.search_text("revenue", similarity_threshold=0.0)
        assert len(results) >= 1
        assert 0.0 <= results[0].similarity <= 1.0

    async def test_tool_and_text_memories_are_independent(self, memory_service):
        await memory_service.store(_make_entry())
        await memory_service.save_text("some text note")
        assert memory_service.count() == 1  # only tool entry counted


# ---------------------------------------------------------------------------
# make_entry_id
# ---------------------------------------------------------------------------


class TestMakeEntryId:
    def test_returns_string(self):
        assert isinstance(make_entry_id(), str)

    def test_unique(self):
        ids = {make_entry_id() for _ in range(100)}
        assert len(ids) == 100
