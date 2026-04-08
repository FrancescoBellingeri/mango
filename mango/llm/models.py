from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tool definition (what the agent exposes to the LLM)
# ---------------------------------------------------------------------------


@dataclass
class ToolParam:
    """A single parameter in a tool definition."""

    name: str
    type: str                           # JSON Schema type: string, integer, object, array, boolean
    description: str
    required: bool = True
    enum: list[str] | None = None       # for string enums
    items: dict[str, Any] | None = None # for array types


@dataclass
class ToolDef:
    """Definition of a tool exposed to the LLM.

    Mirrors the function-calling format used by Anthropic and OpenAI.
    Each tool registered in the ToolRegistry has a corresponding ToolDef.
    """

    name: str
    description: str
    params: list[ToolParam]


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""

    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str = ""              # provider-specific call ID, used in multi-turn
    thought_signature: bytes | None = None  # Gemini thinking models only


@dataclass
class LLMResponse:
    """Response returned by LLMService.chat().

    The LLM either produces a text reply, requests tool calls, or both.
    """

    text: str | None                    # natural language text, if any
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Message format (shared across providers)
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in the conversation history."""

    role: str           # 'user', 'assistant', 'tool'
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None     # only for role='tool'