"""Memory tools exposed to the LLM.

These tools allow the LLM to autonomously save and retrieve interactions
from the agent memory, following the same pattern as Vanna AI's memory system.

Tools defined here:
    - search_saved_correct_tool_uses : search memory for similar past interactions
    - save_question_tool_args        : save a successful (question → tool call) pair
    - save_text_memory               : save free-form text (schema notes, business context)
"""

from __future__ import annotations

from typing import Any

from mango.integrations.chromadb import make_entry_id
from mango.llm import ToolDef, ToolParam
from mango.memory import MemoryEntry, MemoryService
from mango.tools.base import Tool, ToolResult


class SearchSavedCorrectToolUsesTool(Tool):
    """Let the LLM search memory for similar past interactions before answering."""

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="search_saved_correct_tool_uses",
            description=(
                "Search memory for similar past interactions before answering a question. "
                "Call this at the START of every question to find relevant examples "
                "of how similar questions were answered before. "
                "Use the results to guide your tool selection and arguments."
            ),
            params=[
                ToolParam(
                    name="question",
                    type="string",
                    description="The user question to search similar past interactions for.",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            entries = await self._memory.retrieve(
                question=kwargs["question"],
                top_k=5,
                similarity_threshold=0.5,
            )
            if not entries:
                return ToolResult(success=True, data={"results": [], "count": 0})

            results = [
                {
                    "question": e.question,
                    "tool_name": e.tool_name,
                    "tool_args": e.tool_args,
                    "result_summary": e.result_summary,
                    "similarity": round(e.similarity, 2),
                }
                for e in entries
            ]
            return ToolResult(success=True, data={"results": results, "count": len(results)})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class SaveQuestionToolArgsTool(Tool):
    """Let the LLM save a successful (question, tool_name, tool_args) triplet."""

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="save_question_tool_args",
            description=(
                "Save a successful question-tool-argument combination to memory for future reference. "
                "Call this after any tool call that produced a correct and useful result, "
                "so similar questions in future sessions can reuse the same approach. "
                "Do NOT call this if the tool returned an error or an empty result."
            ),
            params=[
                ToolParam(
                    name="question",
                    type="string",
                    description="The original natural language question asked by the user.",
                ),
                ToolParam(
                    name="tool_name",
                    type="string",
                    description="The name of the tool that was used successfully.",
                ),
                ToolParam(
                    name="tool_args",
                    type="object",
                    description="The exact arguments passed to the tool.",
                ),
                ToolParam(
                    name="result_summary",
                    type="string",
                    description="Brief human-readable summary of what the result contained.",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        entry = MemoryEntry(
            id=make_entry_id(),
            question=kwargs["question"],
            tool_name=kwargs["tool_name"],
            tool_args=kwargs["tool_args"],
            result_summary=kwargs["result_summary"][:300],
        )
        try:
            await self._memory.store(entry)
            return ToolResult(success=True, data={"saved_id": entry.id})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class SaveTextMemoryTool(Tool):
    """Let the LLM save free-form text knowledge about the database."""

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="save_text_memory",
            description=(
                "Save free-form text to memory for important insights about the database: "
                "collection purposes, field meanings, business terminology, data quirks. "
                "Use this when you discover something useful that would help answer "
                "future questions (e.g. 'the accounts collection stores customer data', "
                "'status field uses 1=active 2=inactive')."
            ),
            params=[
                ToolParam(
                    name="content",
                    type="string",
                    description="The text content to save.",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            entry_id = await self._memory.save_text(kwargs["content"])
            return ToolResult(success=True, data={"saved_id": entry_id})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
