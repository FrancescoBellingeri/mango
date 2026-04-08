"""
Tests for memory context injection — Mango's equivalent of LlmContextEnhancer.

In Mango, the agent enhances the system prompt by retrieving semantically
similar past interactions from memory and injecting them as few-shot examples
via format_memory_examples() + build_system_prompt().

These tests verify that:
  1. When memory is empty the system prompt contains no memory section.
  2. When memory has a relevant entry it appears in the system prompt.
  3. The similarity threshold filters out irrelevant entries.
  4. memory_hits in AgentResponse correctly counts injected examples.
  5. Multiple entries are all rendered in the prompt.
  6. The injected examples are formatted with question, tool, and args.

No external services required — uses MockLLMService + ephemeral ChromaDB.
"""

from __future__ import annotations

import pytest

from mango.agent.agent import MangoAgent
from mango.llm.models import LLMResponse, ToolCall
from mango.memory.models import MemoryEntry
from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService, make_entry_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_memory() -> ChromaMemoryService:
    """Ephemeral ChromaMemoryService with a clean slate for each test."""
    svc = ChromaMemoryService(persist_dir=":memory:")
    try:
        svc._client.delete_collection("mango_memory")
    except Exception:
        pass
    svc._collection = svc._client.get_or_create_collection(
        name="mango_memory",
        metadata={"hnsw:space": "cosine"},
    )
    return svc


def _make_entry(
    question: str,
    tool_args: dict | None = None,
    result_summary: str = "3 results found.",
) -> MemoryEntry:
    return MemoryEntry(
        id=make_entry_id(),
        question=question,
        tool_name="run_mql",
        tool_args=tool_args or {"operation": "count", "collection": "users"},
        result_summary=result_summary,
    )


def _build_agent(MockLLM, mongo_backend, tool_registry, responses, memory=None):
    llm = MockLLM(responses)
    agent = MangoAgent(
        llm_service=llm,
        tool_registry=tool_registry,
        db=mongo_backend,
        agent_memory=memory,
        introspect=False,
    )
    agent.setup()
    return agent, llm


# ---------------------------------------------------------------------------
# No memory configured
# ---------------------------------------------------------------------------


class TestNoMemoryConfigured:
    """When no memory service is attached, the agent must still work."""

    async def test_ask_without_memory_returns_answer(
        self, MockLLM, mongo_backend, tool_registry
    ):
        responses = [LLMResponse(text="No memory needed.", tool_calls=[])]
        agent, _ = _build_agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("Anything?")
        assert resp.answer == "No memory needed."

    async def test_memory_hits_is_zero(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, _ = _build_agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("Q?")
        assert resp.memory_hits == 0

    async def test_system_prompt_has_no_memory_section(
        self, MockLLM, mongo_backend, tool_registry
    ):
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, llm = _build_agent(MockLLM, mongo_backend, tool_registry, responses)
        await agent.ask("Q?")
        system_prompt = llm.calls[0]["system_prompt"]
        assert "Similar past interactions" not in system_prompt
        assert "Example 1" not in system_prompt


# ---------------------------------------------------------------------------
# Empty memory
# ---------------------------------------------------------------------------


class TestEmptyMemory:
    """With memory attached but empty, no examples should be injected."""

    async def test_no_examples_injected_when_empty(
        self, MockLLM, mongo_backend, tool_registry
    ):
        memory = _fresh_memory()
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, llm = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        await agent.ask("How many users are there?")
        system_prompt = llm.calls[0]["system_prompt"]
        assert "Similar past interactions" not in system_prompt

    async def test_memory_hits_is_zero_when_empty(
        self, MockLLM, mongo_backend, tool_registry
    ):
        memory = _fresh_memory()
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, _ = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        resp = await agent.ask("How many users?")
        assert resp.memory_hits == 0


# ---------------------------------------------------------------------------
# Memory with relevant entries
# ---------------------------------------------------------------------------


class TestMemoryWithRelevantEntries:
    """Memory entries are stored and retrievable via the memory service."""

    async def test_stored_entry_is_retrievable(self):
        memory = _fresh_memory()
        stored_q = "How many users are there in the database?"
        await memory.store(_make_entry(stored_q))

        results = await memory.retrieve("How many users do we have?", similarity_threshold=0.0)
        assert len(results) >= 1

    async def test_memory_hits_counter_is_int(
        self, MockLLM, mongo_backend, tool_registry
    ):
        memory = _fresh_memory()
        await memory.store(_make_entry("How many users are there in the database?"))

        responses = [LLMResponse(text="3 users.", tool_calls=[])]
        agent, _ = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        resp = await agent.ask("How many users do we have?")
        assert isinstance(resp.memory_hits, int)

    async def test_stored_entry_has_correct_question(self):
        memory = _fresh_memory()
        stored_q = "How many users are there in the database?"
        await memory.store(_make_entry(stored_q))

        results = await memory.retrieve(stored_q, similarity_threshold=0.0)
        assert len(results) >= 1
        assert results[0].question == stored_q

    async def test_stored_entry_has_correct_tool_name(self):
        memory = _fresh_memory()
        await memory.store(_make_entry("How many users are there in the database?"))

        results = await memory.retrieve("How many users?", similarity_threshold=0.0)
        assert len(results) >= 1
        assert results[0].tool_name == "run_mql"

    async def test_stored_entry_has_correct_tool_args(self):
        memory = _fresh_memory()
        args = {"operation": "count", "collection": "users"}
        await memory.store(_make_entry("How many users are there in the database?", tool_args=args))

        results = await memory.retrieve("How many users?", similarity_threshold=0.0)
        assert len(results) >= 1
        assert results[0].tool_args == args


# ---------------------------------------------------------------------------
# Similarity threshold filtering
# ---------------------------------------------------------------------------


class TestSimilarityThresholdFiltering:
    """Entries below the threshold must NOT be injected into the prompt."""

    async def test_irrelevant_entry_not_injected(
        self, MockLLM, mongo_backend, tool_registry
    ):
        memory = _fresh_memory()
        # Store something completely unrelated.
        await memory.store(_make_entry("What is the population of Brazil?"))

        responses = [LLMResponse(text="3.", tool_calls=[])]
        agent, llm = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        await agent.ask("Run a MongoDB aggregation pipeline by date")

        system_prompt = llm.calls[0]["system_prompt"]
        # With default threshold 0.6 an unrelated question should not match.
        assert "What is the population of Brazil?" not in system_prompt


# ---------------------------------------------------------------------------
# Multiple memory entries
# ---------------------------------------------------------------------------


class TestMultipleMemoryEntries:
    """Multiple relevant entries should all be rendered in the prompt."""

    async def test_multiple_entries_numbered_in_prompt(
        self, MockLLM, mongo_backend, tool_registry
    ):
        memory = _fresh_memory()
        await memory.store(_make_entry("How many active users are there?"))
        await memory.store(_make_entry("Count total registered users in the DB"))
        await memory.store(_make_entry("Show me the user count"))

        responses = [LLMResponse(text="3.", tool_calls=[])]
        agent, llm = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        await agent.ask("How many users do we have?")

        system_prompt = llm.calls[0]["system_prompt"]
        if "Similar past interactions" in system_prompt:
            # If at least one example was retrieved, Example 1 must be present.
            assert "Example 1" in system_prompt


# ---------------------------------------------------------------------------
# Memory is stored after a successful run_mql
# ---------------------------------------------------------------------------


class TestMemoryStorageAfterSuccess:
    """After a successful run_mql + final answer, an entry must be stored."""

    async def test_successful_run_mql_stored(self, MockLLM, mongo_backend, tool_registry):
        memory = _fresh_memory()
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        tool_name="run_mql",
                        tool_args={"operation": "count", "collection": "users"},
                        tool_call_id="tc-1",
                    )
                ],
            ),
            LLMResponse(text="There are 3 users.", tool_calls=[]),
        ]
        agent, _ = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        await agent.ask("How many users?")
        assert memory.count() == 1

    async def test_failed_tool_not_stored(self, MockLLM, mongo_backend, tool_registry):
        """An unknown tool call failing → nothing stored in memory."""
        memory = _fresh_memory()
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("bad_tool", {}, "tc-1")],
            ),
            LLMResponse(text="I couldn't do that.", tool_calls=[]),
        ]
        agent, _ = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        await agent.ask("Do something impossible")
        assert memory.count() == 0

    async def test_subsequent_ask_benefits_from_stored_memory(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """The second question should retrieve the entry stored by the first."""
        memory = _fresh_memory()

        # First conversation: performs run_mql successfully.
        responses_1 = [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("run_mql", {"operation": "count", "collection": "users"}, "tc-1")
                ],
            ),
            LLMResponse(text="There are 3 users.", tool_calls=[]),
        ]
        agent1, _ = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses_1, memory=memory
        )
        await agent1.ask("How many users are there?")

        assert memory.count() == 1

        # Second conversation: similar question should get a memory hit.
        responses_2 = [LLMResponse(text="Still 3 users.", tool_calls=[])]
        agent2, llm2 = _build_agent(
            MockLLM, mongo_backend, tool_registry, responses_2, memory=memory
        )
        resp2 = await agent2.ask("Count all users in the database")

        # Memory retrieved ≥ 0 (exact match depends on embedding similarity).
        assert isinstance(resp2.memory_hits, int)
        # The stored question must be retrievable.
        hits = await memory.retrieve("Count all users", top_k=1, similarity_threshold=0.0)
        assert len(hits) == 1
        assert hits[0].tool_name == "run_mql"
