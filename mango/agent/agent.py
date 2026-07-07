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

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable

import re

from mango.agent.prompt_builder import (
    build_system_prompt,
    schema_section_for_query,
    value_hints_section,
    _FULL_SCHEMA_THRESHOLD,
)
from mango.agent.value_grounding import ValueIndex, build_value_index, find_value_hints
from mango.nosql_runner import NoSQLRunner
from mango.core.types import SchemaInfo
from mango.llm import LLMService, Message, SystemPromptPart
from mango.memory import MemoryEntry, MemoryService, make_entry_id
from mango.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Tool names that should never be auto-saved to memory.
_MEMORY_TOOL_NAMES: frozenset[str] = frozenset({
    "save_question_tool_args",
    "save_text_memory",
})

# Exception *kinds* (ToolResult.error_kind = exception class name) that indicate
# infrastructure failures the LLM cannot fix by rewriting its query. Everything
# else — validation errors, query errors, execution timeouts, bad tool args,
# returned failures with no kind — is treated as retryable so the LLM gets a
# chance to correct itself.
_FATAL_ERROR_KINDS: frozenset[str] = frozenset({
    "BackendError",
    "LLMError",
    "ConnectionFailure",
    "ServerSelectionTimeoutError",
    "AutoReconnect",
})


def _is_retryable(error_kind: str | None) -> bool:
    """Return True when the error is a query/logic error the LLM can fix.

    Classifies by exception *type*, not by substring-matching the message: a
    collection named 'network_events' or a query that 'timed out' must not be
    misread as an infrastructure failure and blocked from a corrective retry.
    """
    return error_kind not in _FATAL_ERROR_KINDS


# Historical tool-result compaction (§9): tool results from completed turns are
# rewritten to row_count + a few sample rows so a large payload is not re-sent
# verbatim on every subsequent LLM call.
_COMPACT_TOOL_RESULT_THRESHOLD = 600  # only compact tool results longer than this
_COMPACT_SAMPLE_ROWS = 3              # rows kept from a run_mql result
_COMPACT_HEAD_CHARS = 400             # chars kept by the generic (non-row) fallback
_COMPACT_MARKER = "_compacted"        # sentinel marking an already-compacted result


def _retry_message(tool_name: str, tool_args: dict, error: str, attempt: int, max_retries: int) -> str:
    args_str = json.dumps(tool_args, default=str, indent=2)
    return (
        f"[RETRY {attempt}/{max_retries}] Tool '{tool_name}' failed.\n"
        f"Args used:\n{args_str}\n"
        f"Error: {error}\n"
        f"The error is fixable. Rules to follow:\n"
        f"- All MongoDB stage and operator names MUST start with '$' (e.g. '$match', '$unwind', '$group', '$sort', '$project'). Never omit the dollar sign.\n"
        f"- Field names and operator names must NOT be wrapped in extra quotes (e.g. use imdb.rating, not \"imdb.rating\").\n"
        f"Correct the query and call '{tool_name}' again."
    )


def _fatal_message(tool_name: str, error: str) -> str:
    return (
        f"[FATAL] Tool '{tool_name}' failed with an infrastructure error that cannot be retried.\n"
        f"Error: {error}\n"
        f"Do not retry. Report the error to the user."
    )


def _exhausted_message(tool_name: str, error: str, max_retries: int) -> str:
    return (
        f"[MAX RETRIES EXCEEDED] Tool '{tool_name}' has failed {max_retries} times with the same error.\n"
        f"Last error: {error}\n"
        f"Do NOT repeat the same approach. Try a different operation or strategy "
        f"(e.g. use 'find' or 'count' instead of 'aggregate', or simplify the query). "
        f"If no alternative exists, report the failure to the user."
    )


@dataclass
class AgentResponse:
    """Final response returned to the caller after one agent turn."""

    answer: str
    tool_calls_made: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    memory_hits: int = 0
    retries_made: int = 0


class MangoAgent:
    """Orchestrates the LLM ↔ tool loop for a single database session.

    Args:
        llm_service: LLM service to use for generating responses.
        tool_registry: Tool registry populated with the available tools.
        db: Connected NoSQL db (used for schema introspection at
                 setup time).
        agent_memory: Optional memory service. If provided, similar past
                      interactions are injected as few-shot examples.
        schema: Pre-introspected schema (optional; fetched lazily if None).
        introspect: Whether to introspect schema at setup() time.
        max_iterations: Safety cap on tool-call iterations per question.
        max_retries: Max retries for fixable tool errors before giving up.
        memory_top_k: Number of memory examples to retrieve per question.
        max_turns: Number of conversation turns to keep in history.
    """

    def __init__(
        self,
        llm_service: LLMService,
        tool_registry: ToolRegistry,
        db: NoSQLRunner,
        agent_memory: MemoryService | None = None,
        schema: dict[str, SchemaInfo] | None = None,
        introspect: bool = False,
        max_iterations: int = 8,
        max_retries: int = 2,
        memory_top_k: int = 3,
        training_top_k: int = 3,
        max_turns: int = 5,
        schema_top_k: int = 3,
        schema_always_all: int = 4,
    ) -> None:
        self._llm = llm_service
        self._db = db
        self._registry = tool_registry
        self._memory = agent_memory
        self._schema = schema
        self._introspect = introspect
        self._max_iterations = max_iterations
        self._max_retries = max_retries
        self._memory_top_k = memory_top_k
        self._training_top_k = training_top_k
        self._max_turns = max_turns
        self._schema_top_k = schema_top_k
        self._schema_always_all = schema_always_all
        self._system_prompt: str = ""
        self._conversation: list[Message] = []
        self._ready: bool = False
        self._last_memory_entry_id: str | None = None
        self._value_index: ValueIndex | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def llm_service(self) -> LLMService:
        return self._llm

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    @property
    def db(self) -> NoSQLRunner:
        return self._db

    @property
    def agent_memory(self) -> MemoryService | None:
        return self._memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise the agent: introspect schema and build system prompt.

        Call once after connecting the db, before the first ask().
        """
        db_name = getattr(
            getattr(self._db, "_database", None), "name", "unknown"
        )

        if self._schema is None and self._introspect:
            collections = self._db.list_collections()
            logger.info("Introspecting schema for %d collections…", len(collections))
            self._schema = self._db.introspect_schema()

        if self._schema and self._value_index is None:
            self._value_index = build_value_index(self._schema)

        # Schema is injected dynamically per-query in _prepare_turn; omit here.
        self._system_prompt = build_system_prompt(db_name=db_name, schema=None)
        self._ready = True
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
            introspect=False,
            max_iterations=self._max_iterations,
            max_retries=self._max_retries,
            memory_top_k=self._memory_top_k,
            training_top_k=self._training_top_k,
            max_turns=self._max_turns,
            schema_top_k=self._schema_top_k,
            schema_always_all=self._schema_always_all,
        )
        agent._system_prompt = self._system_prompt
        agent._ready = self._ready
        agent._value_index = self._value_index
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
        memory_hits, system_prompt_parts = await self._prepare_turn(question)

        async for event in self._run_loop(question, system_prompt_parts, memory_hits):
            if event["type"] == "tool_result" and on_tool_call:
                on_tool_call(
                    event["tool_name"],
                    event["tool_args"],
                    event["result_text"],
                )
            if event["type"] == "answer":
                self._prune_conversation()
                return AgentResponse(
                    answer=event["text"],
                    tool_calls_made=event["tool_calls_made"],
                    input_tokens=event["input_tokens"],
                    output_tokens=event["output_tokens"],
                    iterations=event["iterations"],
                    memory_hits=event["memory_hits"],
                    retries_made=event["retries_made"],
                )

        # Should never reach here — _run_loop always yields an answer event.
        return AgentResponse(answer="")

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
               "output_tokens": int, "memory_hits": int, "tool_calls_made": list[str],
               "retries_made": int}``
        - ``{"type": "error", "message": str}``
        """
        memory_hits, system_prompt_parts = await self._prepare_turn(question)

        async for event in self._run_loop(question, system_prompt_parts, memory_hits):
            if event["type"] == "tool_call":
                yield {"type": "tool_call", "tool_name": event["tool_name"], "tool_args": event["tool_args"]}
            elif event["type"] == "tool_result":
                yield {
                    "type": "tool_result",
                    "tool_name": event["tool_name"],
                    "success": event["success"],
                    "preview": event["result_text"],
                }
            elif event["type"] == "answer":
                self._prune_conversation()
                yield {"type": "answer", "text": event["text"]}
                yield {
                    "type": "done",
                    "iterations": event["iterations"],
                    "input_tokens": event["input_tokens"],
                    "output_tokens": event["output_tokens"],
                    "memory_hits": event["memory_hits"],
                    "tool_calls_made": event["tool_calls_made"],
                    "retries_made": event["retries_made"],
                }

    def reset_conversation(self) -> None:
        """Clear conversation history (start a new session)."""
        self._conversation = []
        logger.debug("Conversation history cleared.")

    @property
    def conversation_length(self) -> int:
        """Number of messages in the current conversation history."""
        return len(self._conversation)

    # ------------------------------------------------------------------
    # Private: core loop
    # ------------------------------------------------------------------

    @staticmethod
    def _stem(token: str) -> str:
        """Lightweight suffix-stripping normaliser for DB schema/query matching.

        Strips a small set of suffixes ordered from longest to shortest so that
        longer patterns take priority.  Simple plural "s" is handled last and
        only when the result would be at least 3 characters and the token does
        not end in "ss" (to avoid "pass" → "pas").

        This is intentionally not a full morphological stemmer — it only needs
        to be good enough to bridge the gap between query vocabulary and
        collection/field names (e.g. "restaurants" ↔ "restaurant", "cuisines"
        ↔ "cuisine", "transactions" ↔ "transaction").
        """
        for suffix in ("ations", "ation", "ities", "ity"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        # Simple plural: strip trailing "s" unless the word ends in "ss".
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            return token[:-1]
        return token

    async def _select_relevant_collections(self, question: str) -> list[str]:
        """Return names of collections most relevant to the question.

        Scoring (additive):
        - +2  exact token match (query token == collection/field keyword)
        - +2  exact stem match (stemmed token == stemmed keyword)
        - +1  prefix match (one is a prefix of the other, min length 4)
        - +3  bonus when the collection name itself appears as a token or stem
              in the question (collection-name affinity)

        Collections are ranked by score; top *schema_top_k* are returned.
        Ties are broken deterministically (alphabetical order of collection name).
        """
        if not self._schema:
            return []
        all_names = list(self._schema.keys())
        if len(all_names) <= self._schema_always_all:
            return all_names

        raw_tokens = set(re.sub(r"[^\w]", " ", question.lower()).split())
        q_tokens = raw_tokens | {self._stem(t) for t in raw_tokens}

        scores: list[tuple[int, str]] = []
        for name in all_names:
            # Split collection name by underscores, spaces, AND camelCase
            # boundaries so that "listingsAndReviews" → ["listings","and","reviews"].
            raw_name_parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()
            raw_keywords: set[str] = set(re.split(r"[_\s]+", raw_name_parts))
            info = self._schema[name]
            for f in info.fields:
                if "." not in f.path:
                    # Split field names the same way.
                    raw_field_parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", f.name).lower()
                    raw_keywords.update(re.split(r"[_\s]+", raw_field_parts))
            raw_keywords.discard("")
            keywords = raw_keywords | {self._stem(k) for k in raw_keywords}

            # Exact matches (raw and stemmed).
            score = 2 * len(raw_tokens & raw_keywords) + len(
                (q_tokens - raw_tokens) & (keywords - raw_keywords)
            )

            # Prefix matches.
            for qt in q_tokens:
                for kw in keywords:
                    if len(qt) >= 4 and qt != kw and (kw.startswith(qt) or qt.startswith(kw)):
                        score += 1
                        break

            # Collection-name affinity bonus: if any part of the collection
            # name (or its stem) appears directly in the query tokens.
            name_parts = {self._stem(p) for p in re.split(r"[_\s]+", raw_name_parts)} | set(
                re.split(r"[_\s]+", raw_name_parts)
            )
            name_parts.discard("")
            if name_parts & q_tokens:
                score += 3

            scores.append((score, name))

        scores.sort(key=lambda x: (-x[0], x[1]))
        return [name for _, name in scores[: self._schema_top_k]]

    async def _prepare_turn(self, question: str) -> tuple[int, str]:
        """Add user message, retrieve memory, build per-turn system prompt.

        Returns:
            (memory_hits, system_prompt) tuple.
        """
        if not self._ready:
            self.setup()

        # Every existing tool result now belongs to a completed turn — shrink the
        # bulky ones before they are re-sent on this turn's LLM calls.
        self._compact_historical_tool_results()

        self._conversation.append(Message(role="user", content=question))

        memory_hits = 0
        memory_context = ""

        if self._memory is not None:
            sections: list[str] = []

            training_entries = await self._memory.get_training_entries(
                question, top_k=self._training_top_k
            )
            if training_entries:
                lines = [
                    "## VERIFIED TRAINING EXAMPLES — use these directly without additional exploration.\n"
                    "If a training example matches the question, call the tool with those exact args.\n"
                    "Do NOT call describe_collection or search_collection when a training example already covers the question.\n"
                ]
                for e in training_entries:
                    lines.append(f"Q: {e.question}")
                    lines.append(f"Tool: {e.tool_name} | Args: {e.tool_args}")
                    if e.result_summary:
                        lines.append(f"Result: {e.result_summary}")
                    lines.append("")
                sections.append("\n".join(lines))

            entries = await self._memory.retrieve(question, top_k=self._memory_top_k)
            if entries:
                known_collections: set[str] = set(self._schema.keys()) if self._schema else set()
                # Silently drop entries that reference a collection no longer
                # present in the schema — they would mislead the agent.
                if known_collections:
                    entries = [
                        e for e in entries
                        if not (
                            isinstance(e.tool_args, dict)
                            and e.tool_args.get("collection")
                            and e.tool_args["collection"] not in known_collections
                        )
                    ]
                if entries:
                    memory_hits = len(entries)
                    lines = ["## Similar past interactions\n"]
                    for e in entries:
                        lines.append(f"Q: {e.question}")
                        lines.append(f"Tool: {e.tool_name} | Args: {e.tool_args}")
                        lines.append(f"Result: {e.result_summary}\n")
                    sections.append("\n".join(lines))

            if sections:
                memory_context = "\n\n".join(sections) + "\n\n"

        schema_section = ""
        if self._schema:
            relevant = await self._select_relevant_collections(question)
            schema_section = schema_section_for_query(
                self._schema, relevant, total_collections=len(self._schema)
            ) + "\n\n"

        value_hints_text = ""
        if self._value_index:
            hints = find_value_hints(question, self._value_index)
            if hints:
                value_hints_text = value_hints_section(hints) + "\n\n"

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
        dynamic = f"Current datetime: {now}\n\n{memory_context}{schema_section}{value_hints_text}".rstrip()
        return memory_hits, [
            SystemPromptPart(text=self._system_prompt, cacheable=True),
            SystemPromptPart(text=dynamic, cacheable=False),
        ]

    async def _run_loop(
        self,
        question: str,
        system_prompt_parts: list[SystemPromptPart],
        memory_hits: int,
    ) -> AsyncGenerator[dict, None]:
        """Core LLM ↔ tool loop. Yields typed event dicts.

        Event types:
          - tool_call:   {type, tool_name, tool_args}
          - tool_result: {type, tool_name, tool_args, success, result_text}
          - answer:      {type, text, iterations, input_tokens, output_tokens,
                          memory_hits, tool_calls_made, retries_made}
        """
        tool_calls_made: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        iterations = 0
        retry_count = 0
        inspected_collections: set[str] = set()
        pending_memory: MemoryEntry | None = None

        while iterations < self._max_iterations:
            iterations += 1

            response = self._llm.chat(
                messages=self._conversation,
                tools=self._registry.get_definitions(),
                system_prompt_parts=system_prompt_parts,
            )

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            if not response.has_tool_calls:
                answer = response.text or ""
                self._conversation.append(Message(role="assistant", content=answer))
                await self._commit_memory(pending_memory)
                yield {
                    "type": "answer",
                    "text": answer,
                    "iterations": iterations,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "memory_hits": memory_hits,
                    "tool_calls_made": tool_calls_made,
                    "retries_made": retry_count,
                }
                return

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
                # Preserve Gemini 3 thought_signature for round-tripping (required to avoid 400 errors).
                if tc.thought_signature is not None:
                    block["thought_signature"] = tc.thought_signature
                assistant_content.append(block)
            self._conversation.append(
                Message(role="assistant", content=assistant_content)
            )

            for tc in response.tool_calls:
                tool_calls_made.append(tc.tool_name)
                logger.info("Tool call: %s(%s)", tc.tool_name, tc.tool_args)

                if tc.tool_name == "describe_collection":
                    col = tc.tool_args.get("collection")
                    if col:
                        inspected_collections.add(col)

                yield {"type": "tool_call", "tool_name": tc.tool_name, "tool_args": tc.tool_args}

                schema_prefix = ""
                if (
                    tc.tool_name == "run_mql"
                    and self._schema is not None
                    and len(self._schema) > _FULL_SCHEMA_THRESHOLD
                ):
                    col = tc.tool_args.get("collection")
                    if col and col not in inspected_collections:
                        desc = await self._registry.execute("describe_collection", collection=col)
                        if desc.success:
                            inspected_collections.add(col)
                            schema_prefix = f"[AUTO-SCHEMA for '{col}']\n{desc.as_text()}\n\n"
                            logger.debug("Auto-injected schema for collection '%s'", col)

                result = await self._registry.execute(tc.tool_name, **tc.tool_args)
                result_text = schema_prefix + result.as_text()

                logger.debug("Tool result (%s): %.200s…", tc.tool_name, result_text)

                if result.success:
                    if tc.tool_name == "run_mql":
                        pending_memory = MemoryEntry(
                            id=make_entry_id(),
                            question=question,
                            tool_name=tc.tool_name,
                            tool_args=tc.tool_args,
                            result_summary=result_text[:300],
                        )
                    retry_count = 0
                else:
                    error_msg = result.error or result_text
                    retryable = _is_retryable(result.error_kind)
                    if retryable and retry_count < self._max_retries:
                        retry_count += 1
                        logger.info(
                            "Retryable error on '%s' (attempt %d/%d): %s",
                            tc.tool_name, retry_count, self._max_retries, error_msg[:120],
                        )
                        result_text = _retry_message(
                            tc.tool_name, tc.tool_args, error_msg, retry_count, self._max_retries
                        )
                    elif not retryable:
                        logger.warning("Non-retryable error on '%s': %s", tc.tool_name, error_msg[:120])
                        result_text = _fatal_message(tc.tool_name, error_msg)
                    else:
                        logger.warning(
                            "Max retries (%d) exceeded on '%s'.", self._max_retries, tc.tool_name
                        )
                        result_text = _exhausted_message(tc.tool_name, error_msg, self._max_retries)

                yield {
                    "type": "tool_result",
                    "tool_name": tc.tool_name,
                    "tool_args": tc.tool_args,
                    "success": result.success,
                    "result_text": result_text,
                }

                self._conversation.append(
                    Message(role="tool", content=result_text, tool_call_id=tc.tool_call_id)
                )

        # Safety cap reached — ask LLM for a final answer with no tools.
        logger.warning("Max iterations (%d) reached.", self._max_iterations)
        response = self._llm.chat(
            messages=self._conversation,
            tools=[],
            system_prompt_parts=system_prompt_parts,
        )
        answer = response.text or "I reached the maximum number of steps. Please try rephrasing your question."
        self._conversation.append(Message(role="assistant", content=answer))
        await self._commit_memory(pending_memory)
        yield {
            "type": "answer",
            "text": answer,
            "iterations": iterations,
            "input_tokens": total_input_tokens + response.input_tokens,
            "output_tokens": total_output_tokens + response.output_tokens,
            "memory_hits": memory_hits,
            "tool_calls_made": tool_calls_made,
            "retries_made": retry_count,
        }

    def _compact_historical_tool_results(self) -> None:
        """Shrink tool results from completed turns before sending them again.

        Called at the start of each turn, when every existing ``role='tool'``
        message belongs to an already-answered turn. A run_mql result can carry
        up to _max_rows rows of JSON that get re-sent verbatim on every
        subsequent LLM call; the assistant has already summarised what mattered
        into its answer, so we replace the bulky payload with row_count + the
        first few rows. Only the *content* is rewritten — the message and its
        tool_call_id stay in place, so tool_use/tool_result pairing (and the
        API message format) is untouched.

        Trade-off: a follow-up that needs raw historical rows ("the 5th row
        from before") loses them and the agent must re-run the query.

        Set env MANGO_COMPACT_TOOL_RESULTS=0 to disable (for A/B testing).
        """
        if os.getenv("MANGO_COMPACT_TOOL_RESULTS", "1").lower() in ("0", "false", "no"):
            return
        compacted = 0
        saved = 0
        for m in self._conversation:
            if m.role != "tool" or not isinstance(m.content, str):
                continue
            if len(m.content) <= _COMPACT_TOOL_RESULT_THRESHOLD:
                continue
            if _COMPACT_MARKER in m.content:
                continue  # already compacted
            original_len = len(m.content)
            m.content = self._summarize_tool_result(m.content)
            compacted += 1
            saved += original_len - len(m.content)
        if compacted:
            logger.info(
                "Compacted %d historical tool result(s), saved ~%d chars (~%d tokens).",
                compacted, saved, saved // 4,
            )

    @staticmethod
    def _summarize_tool_result(text: str) -> str:
        """Return a compact summary of a large tool-result string."""
        start = text.find("{")
        if start != -1:
            try:
                payload, _ = json.JSONDecoder().raw_decode(text[start:])
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                rows = payload["rows"]
                n = payload.get("row_count", len(rows))
                omitted = max(0, n - _COMPACT_SAMPLE_ROWS)
                summary = {
                    "row_count": n,
                    "sample_rows": rows[:_COMPACT_SAMPLE_ROWS],
                    _COMPACT_MARKER: f"{omitted} more rows omitted; re-run the query for full results",
                }
                return json.dumps(
                    summary, default=str, ensure_ascii=False, separators=(",", ":")
                )
        # Generic fallback: keep the head, flag the omission.
        head = text[:_COMPACT_HEAD_CHARS].rstrip()
        return f"{head}… [{_COMPACT_MARKER}: {len(text) - _COMPACT_HEAD_CHARS} chars of tool output dropped; re-run to see the full result]"

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
        self._conversation = self._conversation[cutoff:]
        logger.debug("Pruned %d messages (%d turns removed).", cutoff, excess)

    async def _commit_memory(self, entry: MemoryEntry | None) -> None:
        """Persist the final run_mql entry to memory. No-op if entry is None or memory disabled."""
        if self._memory is None or entry is None:
            return
        try:
            await self._memory.store(entry)
            self._last_memory_entry_id = entry.id
            logger.info("Auto-saved memory entry: %s(%s)", entry.tool_name, str(entry.tool_args)[:60])
        except Exception as exc:
            logger.warning("Failed to auto-save memory entry: %s", exc)
