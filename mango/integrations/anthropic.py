"""Anthropic Claude implementation of LLMService."""

from __future__ import annotations

from mango.llm import LLMResponse, LLMService, Message, ToolCall, ToolDef, ToolParam


def _build_input_schema(params: list[ToolParam]) -> dict:
    """Convert ToolParam list to JSON Schema object expected by Anthropic."""
    properties: dict = {}
    required: list[str] = []

    for p in params:
        prop: dict = {"type": p.type, "description": p.description}
        if p.enum:
            prop["enum"] = p.enum
        if p.items:
            prop["items"] = p.items
        properties[p.name] = prop
        if p.required:
            required.append(p.name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _to_anthropic_tools(tools: list[ToolDef]) -> list[dict]:
    """Convert ToolDef list to Anthropic tool definitions."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": _build_input_schema(t.params),
        }
        for t in tools
    ]


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    """Convert Message list to Anthropic message format."""
    result = []
    for m in messages:
        if m.role == "tool":
            # Tool results are appended inside a user turn
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content if isinstance(m.content, str) else str(m.content),
                    }
                ],
            })
        else:
            result.append({"role": m.role, "content": m.content})
    return result


class AnthropicLlmService(LLMService):
    """LLMService backed by Anthropic Claude.

    Args:
        api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
        model: Model ID to use. Defaults to claude-sonnet-4-6.
        max_tokens: Max tokens in the response.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is not installed. Run: pip install mango-ai[anthropic]"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str = "",
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": _to_anthropic_messages(messages),
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        response = self._client.messages.create(**kwargs)

        text: str | None = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_name=block.name,
                        tool_args=block.input,
                        tool_call_id=block.id,
                    )
                )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def get_model_name(self) -> str:
        return self._model

