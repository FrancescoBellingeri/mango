"""FastAPI route handlers."""

from __future__ import annotations

import json
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mango.servers.fastapi.models import (
    AskRequest,
    HealthResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", service="mango")


# ---------------------------------------------------------------------------
# Ask (natural language)
# ---------------------------------------------------------------------------

@router.post("/ask/stream")
async def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """Stream agent events via Server-Sent Events (SSE).

    The response is a ``text/event-stream`` where each ``data:`` line is a
    JSON object with a ``type`` field:

    - ``session``     — ``{session_id}`` sent immediately so the client can
                        continue the conversation.
    - ``tool_call``   — ``{tool_name, tool_args}`` when the LLM invokes a tool.
    - ``tool_result`` — ``{tool_name, success, preview}`` after execution.
    - ``answer``      — ``{text}`` the final natural language answer.
    - ``done``        — ``{iterations, input_tokens, output_tokens,
                          memory_hits, tool_calls_made}`` end-of-stream metadata.
    - ``error``       — ``{message}`` if an unhandled exception occurs.
    """
    agent = request.app.state.agent
    session_id = body.session_id or str(uuid4())

    async def _generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        try:
            async for event in agent.ask_stream(body.question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
        },
    )


