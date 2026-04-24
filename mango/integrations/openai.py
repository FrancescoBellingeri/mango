"""OpenAI implementation of LLMService."""

from __future__ import annotations

import json

from mango.llm import LLMResponse, LLMService, Message, ToolCall, ToolDef, ToolParam


class OpenAILlmService(LLMService):
    """LLMService backed by OpenAI.

    Args:
        api_key: OpenAI API key. If None, reads OPENAI_API_KEY from env.
        model: Model ID to use. Defaults to gpt-4o.
        max_completion_tokens: Max tokens in the response.
    """

    DEFAULT_MODEL = "gpt-5.4"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_completion_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is not installed. Run: pip install mango-ai[openai]"
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_completion_tokens = max_completion_tokens

    @staticmethod
    def _build_parameters(params: list[ToolParam]) -> dict:
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

    @staticmethod
    def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": OpenAILlmService._build_parameters(t.params),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _to_openai_messages(messages: list[Message], system_prompt: str) -> list[dict]:
        result: list[dict] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for m in messages:
            if m.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                })
            elif m.role == "assistant" and isinstance(m.content, list):
                # Reconstruct assistant message with tool_calls.
                text_parts = [b["text"] for b in m.content if b.get("type") == "text"]
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]),
                        },
                    }
                    for b in m.content
                    if b.get("type") == "tool_use"
                ]
                msg: dict = {"role": "assistant"}
                if text_parts:
                    msg["content"] = " ".join(text_parts)
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                result.append(msg)
            else:
                result.append({"role": m.role, "content": m.content})

        return result

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str = "",
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_completion_tokens": self._max_completion_tokens,
            "messages": self._to_openai_messages(messages, system_prompt),
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        text: str | None = message.content or None
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        tool_args=args,
                        tool_call_id=tc.id,
                    )
                )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def get_model_name(self) -> str:
        return self._model

