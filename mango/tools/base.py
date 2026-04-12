"""Tool ABC, ToolResult and ToolRegistry.

A Tool is an action the agent can invoke on behalf of the LLM.
The ToolRegistry is the central catalogue that maps tool names to their
implementations and exposes ToolDef lists to the LLMService.

ToolRegistry is also accessible from its canonical public path::

    from mango.core.registry import ToolRegistry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mango.llm import ToolDef


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result returned by a Tool after execution.

    Args:
        success: Whether the tool completed without errors.
        data: The result payload (any JSON-serialisable value).
        error: Human-readable error message when success=False.
    """

    success: bool
    data: Any = None
    error: str | None = None

    def as_text(self) -> str:
        """Render the result as a string to feed back into the LLM."""
        if not self.success:
            return f"ERROR: {self.error}"
        if isinstance(self.data, str):
            return self.data
        import json
        try:
            return json.dumps(self.data, default=str, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(self.data)


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for all agent tools.

    Every tool must declare its ToolDef (name, description, parameters)
    and implement execute() which the agent calls when the LLM requests it.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDef:
        """Return the ToolDef that describes this tool to the LLM."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the arguments provided by the LLM.

        Args:
            **kwargs: Arguments as declared in definition.params.

        Returns:
            ToolResult with success flag and result data or error message.
        """


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Central catalogue of available tools.

    Usage::

        registry = ToolRegistry()
        registry.register(MyTool())

        # Pass tool definitions to the LLMService
        tool_defs = registry.get_definitions()

        # Dispatch a tool call from the LLM
        result = registry.execute("my_tool", arg1="value")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool

    def get_definitions(self) -> list[ToolDef]:
        """Return ToolDef list to pass to LLMService.chat()."""
        return [t.definition for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments.

        Returns a failed ToolResult (instead of raising) for unknown tools,
        so the agent can relay the error back to the LLM gracefully.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool '{tool_name}'. Available: {list(self._tools)}",
            )
        try:
            return await tool.execute(**kwargs)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
