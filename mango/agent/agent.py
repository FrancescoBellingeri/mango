"""Mango agent — the main orchestration loop.

The agent:
  1. Receives a natural language question from the user.
  2. Retrieves similar past interactions from memory (if available).
  3. Sends the question to the LLM with tools, system prompt and memory examples.
  4. Dispatches any tool calls the LLM requests via the ToolRegistry.
  5. Feeds tool results back to the LLM.
  6. Returns the final text answer and optionally stores it in memory.

The loop runs until the LLM produces a text response without tool calls,
or until max_iterations is reached (safety cap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable

from mango.agent.prompt_builder import build_system_prompt
from mango.nosql_runner import NoSQLRunner
from mango.core.types import SchemaInfo
from mango.llm import LLMService, Message
from mango.memory import MemoryEntry, MemoryService
from mango.integrations.chromadb import make_entry_id
from mango.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Tool names that should never be auto-saved to memory.
_MEMORY_TOOL_NAMES: frozenset[str] = frozenset({
    "search_saved_correct_tool_uses",
    "save_question_tool_args",
    "save_text_memory",
})


@dataclass
class AgentResponse:
    """Final response returned to the caller after one agent turn."""

    answer: str
    tool_calls_made: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    memory_hits: int = 0        # number of memory examples injected


class MangoAgent:
    """Orchestrates the LLM ↔ tool loop for a single database session.

    Args:
        llm_service: LLM service to use for generating responses.
        tool_registry: Tool registry populated with the available tools.
        db: Connected NoSQL db (used for schema introspection at
                 setup time). Optional — if None, no schema is injected and
                 setup() is a no-op.
        agent_memory: Optional memory service. If provided, similar past
                      interactions are injected as few-shot examples.
        schema: Pre-introspected schema (optional; fetched lazily if None).
        introspect: Whether to introspect schema at setup() time.
        max_iterations: Safety cap on tool-call iterations per question.
        memory_top_k: Number of memory examples to retrieve per question.
    """

    def __init__(
        self,
        llm_service: LLMService,
        tool_registry: ToolRegistry,
        db: NoSQLRunner | None = None,
        agent_memory: MemoryService | None = None,
        schema: dict[str, SchemaInfo] | None = None,
        introspect: bool = True,
        max_iterations: int = 8,
        memory_top_k: int = 3,
        max_turns: int = 5,
    ) -> None:
        self._llm = llm_service
        self._db = db
        self._registry = tool_registry
        self._memory = agent_memory
        self._schema = schema
        self._introspect = introspect
        self._max_iterations = max_iterations
        self._memory_top_k = memory_top_k
        self._max_turns = max_turns
        self._system_prompt: str = ""
        self._conversation: list[Message] = []

    # ------------------------------------------------------------------
    # Properties (expose internals cleanly for the server layer)
    # ------------------------------------------------------------------

    @property
    def llm_service(self) -> LLMService:
        """The configured LLM service."""
        return self._llm

    @property
    def tool_registry(self) -> ToolRegistry:
        """The tool registry."""
        return self._registry

    @property
    def db(self) -> NoSQLRunner | None:
        """The connected database db (may be None)."""
        return self._db

    @property
    def agent_memory(self) -> MemoryService | None:
        """The memory service (may be None)."""
        return self._memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise the agent: introspect schema and build system prompt.

        Call once after connecting the db, before the first ask().
        If no db was provided, this method is a no-op.
        """
        if self._db is None:
            logger.info("No db configured — skipping schema introspection.")
            return

        db_name = getattr(
            getattr(self._db, "_database", None), "name", "unknown"
        )

        if self._schema is None and self._introspect:
            collections = self._db.list_collections()
            logger.info("Introspecting schema for %d collections…", len(collections))
            self._schema = self._db.introspect_schema()

        self._system_prompt = build_system_prompt(
            db_name=db_name,
            schema=self._schema,
        )
        logger.info("Agent ready. Model: %s", self._llm.get_model_name())

    def new_session(self) -> MangoAgent:
        """Return a new agent with the same configuration but a fresh conversation.

        The schema introspection and system prompt are shared (not re-run),
        making session creation cheap.
        """
        agent = MangoAgent(
            llm_service=self._llm,
            tool_registry=self._registry,
            db=self._db,
            agent_memory=self._memory,
            schema=self._schema,
            introspect=False,   # schema already available
            max_iterations=self._max_iterations,
            memory_top_k=self._memory_top_k,
        )
        # Share the already-built system prompt.
        agent._system_prompt = self._system_prompt
        return agent

    async def ask(
        self,
        question: str,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> AgentResponse:
        """Ask the agent a natural language question.

        The conversation history is preserved across calls so follow-up
        questions work naturally ("and how many were created last month?").

        Args:
            question: Natural language question from the user.
            on_tool_call: Optional callback invoked after each tool execution.
                Receives (tool_name, tool_args, result_text).

        Returns:
            AgentResponse with the answer and metadata.
        """
        if not self._system_prompt:
            self.setup()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
        system_prompt = f"Current datetime: {now}\n\n{self._system_prompt}"
        self._conversation.append(Message(role="user", content=question))

        tool_calls_made: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        iterations = 0

        while iterations < self._max_iterations:
            iterations += 1

            response = self._llm.chat(
                messages=self._conversation,
                tools=self._registry.get_definitions(),
                system_prompt=system_prompt,
            )

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            # If the LLM returned text with no tool calls, we're done.
            if not response.has_tool_calls:
                answer = response.text or ""
                self._conversation.append(
                    Message(role="assistant", content=answer)
                )
                self._prune_conversation()
                return AgentResponse(
                    answer=answer,
                    tool_calls_made=tool_calls_made,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    iterations=iterations,
                    memory_hits=0,
                )

            # Record the assistant turn (may include both text and tool calls).
            assistant_content: list[dict] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                block: dict = {
                    "type": "tool_use",
                    "id": tc.tool_call_id,
                    "name": tc.tool_name,
                    "input": tc.tool_args,
                }
                # Preserve Gemini 3 thought_signature so it can be round-tripped
                # back on subsequent turns (required to avoid 400 errors).
                if tc.thought_signature is not None:
                    block["thought_signature"] = tc.thought_signature
                assistant_content.append(block)
            self._conversation.append(
                Message(role="assistant", content=assistant_content)
            )

            # Execute each tool call and append results.
            for tc in response.tool_calls:
                tool_calls_made.append(tc.tool_name)
                logger.info("Tool call: %s(%s)", tc.tool_name, tc.tool_args)

                result = await self._registry.execute(tc.tool_name, **tc.tool_args)
                result_text = result.as_text()

                logger.debug("Tool result (%s): %.200s…", tc.tool_name, result_text)

                if result.success and tc.tool_name not in _MEMORY_TOOL_NAMES:
                    await self._auto_store_memory(question, tc.tool_name, tc.tool_args, result_text)

                if on_tool_call:
                    on_tool_call(tc.tool_name, tc.tool_args, result_text)

                self._conversation.append(
                    Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=tc.tool_call_id,
                    )
                )

        # Safety cap reached — ask LLM for a final answer with what it has.
        logger.warning("Max iterations (%d) reached.", self._max_iterations)
        response = self._llm.chat(
            messages=self._conversation,
            tools=[],
            system_prompt=system_prompt,
        )
        answer = response.text or "I reached the maximum number of steps. Please try rephrasing your question."
        self._conversation.append(Message(role="assistant", content=answer))
        return AgentResponse(
            answer=answer,
            tool_calls_made=tool_calls_made,
            input_tokens=total_input_tokens + response.input_tokens,
            output_tokens=total_output_tokens + response.output_tokens,
            iterations=iterations,
            memory_hits=0,
        )

    async def ask_stream(
        self,
        question: str,
    ) -> AsyncGenerator[dict, None]:
        """Stream agent events as they happen via a generator.

        Yields dicts with a ``type`` key:

        - ``{"type": "tool_call", "tool_name": str, "tool_args": dict}``
        - ``{"type": "tool_result", "tool_name": str, "success": bool, "preview": str}``
        - ``{"type": "answer", "text": str}``
        - ``{"type": "done", "iterations": int, "input_tokens": int,
               "output_tokens": int, "memory_hits": int, "tool_calls_made": list[str]}``
        - ``{"type": "error", "message": str}``
        """
        if not self._system_prompt:
            self.setup()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
        system_prompt = f"Current datetime: {now}\n\n{self._system_prompt}"
        self._conversation.append(Message(role="user", content=question))

        tool_calls_made: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        iterations = 0

        while iterations < self._max_iterations:
            iterations += 1

            response = self._llm.chat(
                messages=self._conversation,
                tools=self._registry.get_definitions(),
                system_prompt=system_prompt,
            )

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            if not response.has_tool_calls:
                answer = response.text or ""
                self._conversation.append(Message(role="assistant", content=answer))
                self._prune_conversation()

                yield {"type": "answer", "text": answer}
                yield {
                    "type": "done",
                    "iterations": iterations,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "memory_hits": 0,
                    "tool_calls_made": tool_calls_made,
                }
                return

            assistant_content: list[dict] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                block: dict = {
                    "type": "tool_use",
                    "id": tc.tool_call_id,
                    "name": tc.tool_name,
                    "input": tc.tool_args,
                }
                if tc.thought_signature is not None:
                    block["thought_signature"] = tc.thought_signature
                assistant_content.append(block)
            self._conversation.append(
                Message(role="assistant", content=assistant_content)
            )

            for tc in response.tool_calls:
                tool_calls_made.append(tc.tool_name)
                logger.info("Tool call (stream): %s(%s)", tc.tool_name, tc.tool_args)

                yield {"type": "tool_call", "tool_name": tc.tool_name, "tool_args": tc.tool_args}

                result = await self._registry.execute(tc.tool_name, **tc.tool_args)
                result_text = result.as_text()

                if result.success and tc.tool_name not in _MEMORY_TOOL_NAMES:
                    await self._auto_store_memory(question, tc.tool_name, tc.tool_args, result_text)

                preview = result_text[:200] + "…" if len(result_text) > 200 else result_text
                yield {
                    "type": "tool_result",
                    "tool_name": tc.tool_name,
                    "success": result.success,
                    "preview": preview,
                }

                self._conversation.append(
                    Message(role="tool", content=result_text, tool_call_id=tc.tool_call_id)
                )

        # Safety cap reached.
        logger.warning("Max iterations (%d) reached.", self._max_iterations)
        response = self._llm.chat(
            messages=self._conversation,
            tools=[],
            system_prompt=system_prompt,
        )
        answer = response.text or "I reached the maximum number of steps. Please try rephrasing your question."
        self._conversation.append(Message(role="assistant", content=answer))
        yield {"type": "answer", "text": answer}
        yield {
            "type": "done",
            "iterations": iterations,
            "input_tokens": total_input_tokens + response.input_tokens,
            "output_tokens": total_output_tokens + response.output_tokens,
            "memory_hits": 0,
            "tool_calls_made": tool_calls_made,
        }

    def _prune_conversation(self) -> None:
        """Remove oldest turns when conversation exceeds max_turns.

        A turn is defined as: user message + all tool calls/results + assistant answer.
        Pruning always removes complete turns to avoid breaking the API message format
        (a tool_result without its tool_use would cause a 400 error).
        """
        turn_starts = [
            i for i, m in enumerate(self._conversation) if m.role == "user"
        ]
        excess = len(turn_starts) - self._max_turns
        if excess <= 0:
            return
        cutoff = turn_starts[excess]
        removed = cutoff
        self._conversation = self._conversation[cutoff:]
        logger.debug("Pruned %d messages (%d turns removed).", removed, excess)

    async def _auto_store_memory(
        self, question: str, tool_name: str, tool_args: dict, result_text: str
    ) -> None:
        """Automatically persist a successful tool call to memory."""
        if self._memory is None:
            return
        entry = MemoryEntry(
            id=make_entry_id(),
            question=question,
            tool_name=tool_name,
            tool_args=tool_args,
            result_summary=result_text[:300],
        )
        try:
            await self._memory.store(entry)
            logger.info("Auto-saved memory entry: %s(%s)", tool_name, str(tool_args)[:60])
        except Exception as exc:
            logger.warning("Failed to auto-save memory entry: %s", exc)

    def reset_conversation(self) -> None:
        """Clear conversation history (start a new session)."""
        self._conversation = []
        logger.debug("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of messages in the current conversation history."""
        return len(self._conversation)

