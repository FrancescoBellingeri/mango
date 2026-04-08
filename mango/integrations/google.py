
"""Google Gemini implementation of LLMService.

Supports both Google AI Studio (api_key) and Vertex AI (vertexai=True).

Google AI Studio:
    GeminiService(api_key="YOUR_KEY", model="gemini-2.5-pro-preview-05-06")

Vertex AI:
    GeminiService(vertexai=True, model="gemini-3.1-pro-preview")
    # reads GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION from env,
    # or uses Application Default Credentials.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mango.llm import LLMResponse, LLMService, Message, ToolCall, ToolDef, ToolParam

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion helpers  (types module passed in to keep imports lazy)
# ---------------------------------------------------------------------------


def _build_schema(params: list[ToolParam], types: Any) -> Any:
    """Convert ToolParam list to a Gemini Schema object."""
    properties: dict = {}
    required: list[str] = []

    for p in params:
        type_map = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
        }
        gemini_type = type_map.get(p.type, "STRING")

        prop_kwargs: dict[str, Any] = {
            "type": gemini_type,
            "description": p.description,
        }
        if p.enum:
            prop_kwargs["enum"] = p.enum
        if p.type == "array" and p.items:
            item_type = type_map.get(p.items.get("type", "string"), "STRING")
            prop_kwargs["items"] = types.Schema(type=item_type)

        properties[p.name] = types.Schema(**prop_kwargs)
        if p.required:
            required.append(p.name)

    return types.Schema(
        type="OBJECT",
        properties=properties,
        required=required if required else None,
    )


def _to_gemini_tools(tools: list[ToolDef], types: Any) -> list:
    """Convert ToolDef list to a single Gemini Tool with all function declarations."""
    if not tools:
        return []

    declarations = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=_build_schema(t.params, types) if t.params else None,
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _to_gemini_contents(
    messages: list[Message],
    system_prompt: str,
    types: Any,
) -> tuple[str | None, list]:
    """Convert Message list to (system_instruction, contents).

    Gemini separates the system prompt from the conversation history.
    Tool results use role 'user' with a FunctionResponse part.
    """
    contents: list = []

    for m in messages:
        if m.role == "tool":
            # Tool result — must be wrapped in a user turn with FunctionResponse.
            part = types.Part.from_function_response(
                name=_extract_tool_name(m),
                response={"result": m.content if isinstance(m.content, str) else json.dumps(m.content)},
            )
            contents.append(types.Content(role="user", parts=[part]))

        elif m.role == "assistant":
            if isinstance(m.content, list):
                # Assistant turn with possible tool calls.
                parts: list = []
                for block in m.content:
                    if block.get("type") == "text":
                        parts.append(types.Part.from_text(text=block["text"]))
                    elif block.get("type") == "tool_use":
                        fc_part = types.Part.from_function_call(
                            name=block["name"],
                            args=block["input"],
                        )
                        # Gemini 3 requires thought_signature to be round-tripped
                        # back on each functionCall part (400 error if missing).
                        if block.get("thought_signature"):
                            fc_part.thought_signature = block["thought_signature"]
                        parts.append(fc_part)
                contents.append(types.Content(role="model", parts=parts))
            else:
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=str(m.content))],
                    )
                )

        else:  # user
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)],
                )
            )

    return system_prompt or None, contents


def _extract_tool_name(m: Message) -> str:
    """Best-effort extraction of the tool name from a tool-result message.

    The tool_call_id in our system is the provider's call ID. For Gemini
    we store the tool name as the tool_call_id (see below).
    """
    return m.tool_call_id or "unknown_tool"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class GeminiLlmService(LLMService):
    """LLMService backed by Google Gemini.

    Args:
        api_key: Google AI Studio API key. If None, reads GOOGLE_API_KEY from env.
                 Not needed when vertexai=True.
        model: Model ID. Defaults to gemini-2.5-pro-preview-05-06.
        vertexai: If True, uses Vertex AI instead of Google AI Studio.
                  Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION from env.
        max_output_tokens: Max tokens in the response.
        temperature: Sampling temperature.
    """

    DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        vertexai: bool = False,
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is not installed. Run: pip install mango-ai[gemini]"
            )
        self._types = types

        if vertexai:
            self._client = genai.Client(vertexai=True)
        else:
            key = api_key or os.getenv("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key)

        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str = "",
    ) -> LLMResponse:
        types = self._types
        system_instruction, contents = _to_gemini_contents(messages, system_prompt, types)

        config_kwargs: dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = _to_gemini_tools(tools, types)

        config = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        text: str | None = None
        tool_calls: list[ToolCall] = []

        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if part.text:
                    text = (text or "") + part.text
                elif part.function_call:
                    fc = part.function_call
                    # Use tool name as tool_call_id so we can reconstruct
                    # the FunctionResponse in the next turn.
                    # Capture thought_signature (Gemini 3 requirement).
                    tool_calls.append(
                        ToolCall(
                            tool_name=fc.name,
                            tool_args=dict(fc.args) if fc.args else {},
                            tool_call_id=fc.name,
                            thought_signature=part.thought_signature or None,
                        )
                    )

        usage = response.usage_metadata
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=self._model,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )

    def get_model_name(self) -> str:
        return self._model
