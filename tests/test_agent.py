"""Tests for mango.agent.agent — MangoAgent orchestration loop."""

from __future__ import annotations

from mango.agent.agent import AgentResponse, MangoAgent
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
        fatal_result = ToolResult(success=False, error="Connection refused: cannot connect to MongoDB")
        with patch.object(agent._registry, "execute", new=AsyncMock(return_value=fatal_result)):
            resp = await agent.ask("?")

        tool_msgs = [m for m in agent._conversation if m.role == "tool"]
        assert any("[FATAL]" in str(m.content) for m in tool_msgs)
        assert resp.retries_made == 0

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
