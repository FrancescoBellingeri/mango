"""Ollama implementation of LLMService (local models)."""

from __future__ import annotations

import json

from mango.llm import LLMResponse, LLMService, Message, ToolCall, ToolDef, ToolParam


class OllamaLlmService(LLMService):
    """LLMService backed by Ollama for local model inference.

    Args:
        model: Ollama model name (e.g. 'llama3.2', 'mistral', 'qwen2.5').
        host: Ollama server URL. Defaults to http://localhost:11434.
        max_tokens: Max tokens in the response.
    """

    DEFAULT_MODEL = "llama3.2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package is not installed. Run: pip install ollama"
            )
        self._ollama = ollama
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _build_tool_schema(params: list[ToolParam]) -> dict:
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
    def _to_ollama_tools(tools: list[ToolDef]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": OllamaLlmService._build_tool_schema(t.params),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _to_ollama_messages(messages: list[Message], system_prompt: str) -> list[dict]:
        result: list[dict] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for m in messages:
            if m.role == "tool":
                result.append({
                    "role": "tool",
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                })
            elif m.role == "assistant" and isinstance(m.content, list):
                text_parts = [b["text"] for b in m.content if b.get("type") == "text"]
                tool_calls = [
                    {
                        "function": {
                            "name": b["name"],
                            "arguments": b["input"],
                        },
                    }
                    for b in m.content
                    if b.get("type") == "tool_use"
                ]
                msg: dict = {"role": "assistant", "content": " ".join(text_parts)}
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
            "messages": self._to_ollama_messages(messages, system_prompt),
            "options": {"num_predict": self._max_tokens},
        }
        if tools:
            kwargs["tools"] = self._to_ollama_tools(tools)

        response = self._client.chat(**kwargs)
        message = response.message

        text: str | None = message.content or None
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for i, tc in enumerate(message.tool_calls):
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        tool_args=args,
                        tool_call_id=f"ollama-tc-{i}",
                    )
                )

        usage = response.usage if hasattr(response, "usage") and response.usage else None
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=self._model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    def get_model_name(self) -> str:
        return self._model
