"""Tests for mango.agent.agent — MangoAgent orchestration loop."""

from __future__ import annotations

from mango.agent.agent import AgentResponse, MangoAgent
from mango.core.types import FieldInfo, SchemaInfo
from mango.llm.models import LLMResponse, ToolCall
from mango.memory.models import MemoryEntry
from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService, make_entry_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(MockLLM, mongo_backend, tool_registry, responses, memory=None):
    """Build a MangoAgent with a MockLLMService pre-configured with responses."""
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


def _fresh_memory() -> ChromaMemoryService:
    """Return a ChromaMemoryService with a guaranteed-empty collection."""
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


# ---------------------------------------------------------------------------
# Direct answer (no tool calls)
# ---------------------------------------------------------------------------


class TestAgentDirectAnswer:
    async def test_returns_llm_text(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="There are 3 users in the database.", tool_calls=[])
        ]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("How many users are there?")
        assert isinstance(resp, AgentResponse)
        assert resp.answer == "There are 3 users in the database."

    async def test_iterations_is_one(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("Any question?")
        assert resp.iterations == 1

    async def test_tool_calls_made_is_empty(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("Any question?")
        assert resp.tool_calls_made == []

    async def test_llm_receives_system_prompt(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="Answer.", tool_calls=[])]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        await agent.ask("Test")
        assert llm.calls[0]["system_prompt"] != ""

    async def test_conversation_history_preserved(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="First answer.", tool_calls=[]),
            LLMResponse(text="Second answer.", tool_calls=[]),
        ]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        await agent.ask("First question")
        await agent.ask("Follow-up question")
        # Second call includes the first exchange in history.
        second_call_messages = llm.calls[1]["messages"]
        assert len(second_call_messages) >= 3  # user + assistant + user


# ---------------------------------------------------------------------------
# Tool call round-trip
# ---------------------------------------------------------------------------


class TestAgentToolCall:
    async def test_single_tool_call_dispatched(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            # First: LLM asks to list collections
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        tool_name="list_collections",
                        tool_args={},
                        tool_call_id="tc-1",
                    )
                ],
            ),
            # Second: LLM gives final answer
            LLMResponse(text="The collections are: users, orders.", tool_calls=[]),
        ]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("What collections exist?")
        assert resp.answer == "The collections are: users, orders."
        assert "list_collections" in resp.tool_calls_made

    async def test_iterations_count(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("list_collections", {}, "tc-1")],
            ),
            LLMResponse(text="Done.", tool_calls=[]),
        ]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("?")
        assert resp.iterations == 2

    async def test_on_tool_call_callback_invoked(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("list_collections", {}, "tc-1")],
            ),
            LLMResponse(text="Done.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)

        calls_received = []

        def on_tool_call(name, args, result_text):
            calls_received.append((name, args, result_text))

        await agent.ask("?", on_tool_call=on_tool_call)
        assert len(calls_received) == 1
        assert calls_received[0][0] == "list_collections"

    async def test_unknown_tool_does_not_crash(self, MockLLM, mongo_backend, tool_registry):
        """Registry handles unknown tools gracefully — agent should not raise."""
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("nonexistent_tool", {}, "tc-1")],
            ),
            LLMResponse(text="I got an error but recovered.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("?")
        assert resp.answer == "I got an error but recovered."


# ---------------------------------------------------------------------------
# reset_conversation
# ---------------------------------------------------------------------------


class TestAgentReset:
    async def test_reset_clears_history(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="Answer.", tool_calls=[]),
            LLMResponse(text="Answer 2.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        await agent.ask("First question")
        assert agent.conversation_length > 0
        agent.reset_conversation()
        assert agent.conversation_length == 0


# ---------------------------------------------------------------------------
# Memory integration
# ---------------------------------------------------------------------------


class TestAgentMemory:
    async def test_successful_run_mql_stored_in_memory(
        self, MockLLM, mongo_backend, tool_registry
    ):
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
        agent, _ = _agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        await agent.ask("How many users?")
        assert memory.count() == 1

    async def test_intermediate_tools_not_stored(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """describe_collection + run_mql → only the run_mql entry saved."""
        memory = _fresh_memory()
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("describe_collection", {"collection": "users"}, "tc-1")],
            ),
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        tool_name="run_mql",
                        tool_args={"operation": "count", "collection": "users"},
                        tool_call_id="tc-2",
                    )
                ],
            ),
            LLMResponse(text="3 users.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses, memory=memory)
        await agent.ask("How many users?")
        assert memory.count() == 1

    async def test_only_last_run_mql_stored(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """Multiple run_mql calls → only the last successful one is stored."""
        memory = _fresh_memory()
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("run_mql", {"operation": "find", "collection": "users"}, "tc-1")
                ],
            ),
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("run_mql", {"operation": "count", "collection": "users"}, "tc-2")
                ],
            ),
            LLMResponse(text="3 users.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses, memory=memory)
        await agent.ask("How many users?")
        assert memory.count() == 1
        # The stored entry is the last run_mql (count), not the first (find).
        stored = memory._collection.get()
        assert stored["metadatas"][0]["tool_name"] == "run_mql"
        import json
        assert json.loads(stored["metadatas"][0]["tool_args"])["operation"] == "count"

    async def test_failed_run_mql_not_stored(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """A failing run_mql (invalid collection) produces no memory entry."""
        memory = _fresh_memory()
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("run_mql", {"operation": "find", "collection": "ghost"}, "tc-1")
                ],
            ),
            LLMResponse(text="Could not find data.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses, memory=memory)
        await agent.ask("Find stuff?")
        assert memory.count() == 0

    async def test_memory_hits_reported(self, MockLLM, mongo_backend, tool_registry):
        memory = _fresh_memory()
        # Pre-populate memory with a similar entry.
        entry = MemoryEntry(
            id=make_entry_id(),
            question="Count total users in the database",
            tool_name="run_mql",
            tool_args={"operation": "count", "collection": "users"},
            result_summary="3 users found.",
        )
        await memory.store(entry)

        responses = [LLMResponse(text="3 users.", tool_calls=[])]
        agent, _ = _agent(
            MockLLM, mongo_backend, tool_registry, responses, memory=memory
        )
        resp = await agent.ask("How many users are there?")
        # memory_hits reflects how many examples were retrieved.
        assert isinstance(resp.memory_hits, int)


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    async def test_retries_made_zero_on_clean_run(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="All good.", tool_calls=[])]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        resp = await agent.ask("?")
        assert resp.retries_made == 0

    async def test_retryable_error_increments_retry_count(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """A run_mql failure on a bad collection triggers retry path."""
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("run_mql", {"operation": "find", "collection": "ghost"}, "tc-1")],
            ),
            LLMResponse(text="Sorry, could not find data.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent._max_retries = 2
        resp = await agent.ask("Find stuff in ghost?")
        # Validator blocks ghost collection → retryable → retries_made == 1
        assert resp.retries_made == 1

    async def test_retry_message_injected_into_conversation(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """The tool result message fed back to LLM should contain RETRY marker."""
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("run_mql", {"operation": "find", "collection": "ghost"}, "tc-1")],
            ),
            LLMResponse(text="Done.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent._max_retries = 2
        await agent.ask("?")
        # The second LLM call receives the retry message in conversation history.
        tool_result_msgs = [
            m for m in agent._conversation if m.role == "tool"
        ]
        assert any("[RETRY" in str(m.content) for m in tool_result_msgs)

    async def test_non_retryable_error_uses_fatal_message(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """Infrastructure errors → [FATAL] in result text, retry_count unchanged."""
        from unittest.mock import AsyncMock, patch
        from mango.tools.base import ToolResult

        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("run_mql", {"operation": "find", "collection": "users"}, "tc-1")],
            ),
            LLMResponse(text="Cannot connect.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        # Fatal is classified by exception *kind* (error_kind), not by message
        # substring — the registry stamps type(exc).__name__ on the ToolResult.
        fatal_result = ToolResult(
            success=False,
            error="Connection refused: cannot connect to MongoDB",
            error_kind="BackendError",
        )
        with patch.object(agent._registry, "execute", new=AsyncMock(return_value=fatal_result)):
            resp = await agent.ask("?")

        tool_msgs = [m for m in agent._conversation if m.role == "tool"]
        assert any("[FATAL]" in str(m.content) for m in tool_msgs)
        assert resp.retries_made == 0

    async def test_substring_looking_error_still_retryable_by_kind(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """A message containing 'network'/'timed out' must NOT be misread as fatal.

        Regression for the old substring classifier: a collection named
        'network_events' or a query that 'timed out' is retryable — the LLM can
        act on the 'did you mean' suggestion or simplify the query.
        """
        from unittest.mock import AsyncMock, patch
        from mango.tools.base import ToolResult

        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("run_mql", {"operation": "find", "collection": "network_events"}, "tc-1")],
            ),
            LLMResponse(text="Recovered.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent._max_retries = 2
        retryable_result = ToolResult(
            success=False,
            error="Collection 'network_events' does not exist. Did you mean 'events'? Query timed out.",
            error_kind="ValidationError",
        )
        with patch.object(agent._registry, "execute", new=AsyncMock(return_value=retryable_result)):
            resp = await agent.ask("?")

        tool_msgs = [m for m in agent._conversation if m.role == "tool"]
        assert any("[RETRY" in str(m.content) for m in tool_msgs)
        assert not any("[FATAL]" in str(m.content) for m in tool_msgs)
        assert resp.retries_made == 1

    async def test_retry_count_resets_after_success(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """After a successful tool call, retry_count resets to 0."""
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("run_mql", {"operation": "find", "collection": "ghost"}, "tc-1")],
            ),
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("list_collections", {}, "tc-2")],
            ),
            LLMResponse(text="Done.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent._max_retries = 2
        resp = await agent.ask("?")
        # First call fails (retryable → retry_count=1), second succeeds → reset → final is 0
        assert resp.retries_made == 0

    async def test_max_retries_exceeded_keeps_original_error(
        self, MockLLM, mongo_backend, tool_registry
    ):
        """When retry cap is hit, original error (not RETRY marker) goes to LLM."""
        from unittest.mock import AsyncMock, patch
        from mango.tools.base import ToolResult

        responses = [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("run_mql", {"operation": "find", "collection": "users"}, "tc-1")],
            ),
            LLMResponse(text="Still failing.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent._max_retries = 0  # cap at 0 → immediately exceeded
        bad_result = ToolResult(success=False, error="Bad query syntax")
        with patch.object(agent._registry, "execute", new=AsyncMock(return_value=bad_result)):
            await agent.ask("?")

        tool_msgs = [m for m in agent._conversation if m.role == "tool"]
        assert any("[MAX RETRIES EXCEEDED]" in str(m.content) for m in tool_msgs)


# ---------------------------------------------------------------------------
# Value-grounding hints wired into the per-turn system prompt
# ---------------------------------------------------------------------------


class TestAgentValueGrounding:
    def _schema_with_status_drift(self) -> dict[str, SchemaInfo]:
        return {
            "orders": SchemaInfo(
                collection_name="orders",
                document_count=10,
                fields=[
                    FieldInfo(
                        name="status", path="status", types=["string"], frequency=1.0,
                        sample_values=["ACTIVE", "inactive"],
                    ),
                ],
                indexes=[],
                sample_documents=[],
            )
        }

    async def test_value_hint_injected_when_question_matches_drifted_value(
        self, MockLLM, mongo_backend, tool_registry
    ):
        llm = MockLLM([LLMResponse(text="Answer.", tool_calls=[])])
        agent = MangoAgent(
            llm_service=llm,
            tool_registry=tool_registry,
            db=mongo_backend,
            schema=self._schema_with_status_drift(),
            introspect=False,
        )
        agent.setup()
        await agent.ask("how many active orders are there?")
        system_prompt = llm.calls[0]["system_prompt"]
        assert "## Value hints for this question" in system_prompt
        assert "ACTIVE" in system_prompt

    async def test_no_value_hint_when_question_has_no_match(
        self, MockLLM, mongo_backend, tool_registry
    ):
        llm = MockLLM([LLMResponse(text="Answer.", tool_calls=[])])
        agent = MangoAgent(
            llm_service=llm,
            tool_registry=tool_registry,
            db=mongo_backend,
            schema=self._schema_with_status_drift(),
            introspect=False,
        )
        agent.setup()
        await agent.ask("what is the total revenue?")
        system_prompt = llm.calls[0]["system_prompt"]
        assert "## Value hints for this question" not in system_prompt

    async def test_value_index_shared_across_new_session(
        self, MockLLM, mongo_backend, tool_registry
    ):
        llm = MockLLM([])
        agent = MangoAgent(
            llm_service=llm,
            tool_registry=tool_registry,
            db=mongo_backend,
            schema=self._schema_with_status_drift(),
            introspect=False,
        )
        agent.setup()
        session = agent.new_session()
        assert session._value_index == agent._value_index
        assert session._value_index is not None


# ---------------------------------------------------------------------------
# Historical tool-result compaction (§9)
# ---------------------------------------------------------------------------


def _big_run_mql_content(n_rows: int = 100) -> str:
    import json
    rows = [
        {"_id": f"id{i:06d}", "name": f"Product {i}", "price": i * 1.5,
         "tier": "gold", "active": True}
        for i in range(n_rows)
    ]
    return json.dumps(
        {"rows": rows, "row_count": n_rows},
        separators=(",", ":"),
    )


class TestHistoricalCompaction:
    def test_large_run_mql_result_compacted(self, MockLLM, mongo_backend, tool_registry):
        import json
        from mango.llm.models import Message

        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, [])
        big = _big_run_mql_content(100)
        agent._conversation = [
            Message(role="user", content="q1"),
            Message(role="assistant", content=[{"type": "tool_use", "id": "tc1", "name": "run_mql", "input": {}}]),
            Message(role="tool", content=big, tool_call_id="tc1"),
            Message(role="assistant", content="answer 1"),
        ]
        agent._compact_historical_tool_results()

        tool_msg = agent._conversation[2]
        # Much smaller, pairing preserved.
        assert len(tool_msg.content) < len(big) * 0.2
        assert tool_msg.tool_call_id == "tc1"
        payload = json.loads(tool_msg.content)
        assert payload["row_count"] == 100
        assert len(payload["sample_rows"]) == 3
        assert "_compacted" in payload

    def test_small_result_untouched(self, MockLLM, mongo_backend, tool_registry):
        from mango.llm.models import Message

        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, [])
        small = '{"rows":[{"count":42}],"row_count":1}'
        agent._conversation = [Message(role="tool", content=small, tool_call_id="tc1")]
        agent._compact_historical_tool_results()
        assert agent._conversation[0].content == small

    def test_idempotent(self, MockLLM, mongo_backend, tool_registry):
        from mango.llm.models import Message

        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, [])
        agent._conversation = [Message(role="tool", content=_big_run_mql_content(100), tool_call_id="tc1")]
        agent._compact_historical_tool_results()
        once = agent._conversation[0].content
        agent._compact_historical_tool_results()
        assert agent._conversation[0].content == once

    def test_env_var_disables_compaction(self, MockLLM, mongo_backend, tool_registry, monkeypatch):
        from mango.llm.models import Message

        monkeypatch.setenv("MANGO_COMPACT_TOOL_RESULTS", "0")
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, [])
        big = _big_run_mql_content(100)
        agent._conversation = [Message(role="tool", content=big, tool_call_id="tc1")]
        agent._compact_historical_tool_results()
        assert agent._conversation[0].content == big  # untouched

    def test_generic_fallback_for_non_row_payload(self):
        import json
        # A large payload without a "rows" list (e.g. a describe schema dump).
        big = json.dumps({"fields": [{"path": f"f{i}", "types": ["str"]} for i in range(200)]})
        out = MangoAgent._summarize_tool_result(big)
        assert len(out) < len(big)
        assert "_compacted" in out
