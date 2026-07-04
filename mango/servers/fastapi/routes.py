"""FastAPI route handlers."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from mango.memory import make_entry_id
from mango.memory.models import TrainingEntry
from mango.servers.fastapi.models import (
    AskRequest,
    ExportResponse,
    HealthResponse,
    TrainRequest,
    TrainResponse,
)

logger = logging.getLogger(__name__)

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
    - ``error``       — ``{message, request_id}`` if an unhandled exception
                        occurs. ``message`` is generic; the full detail is
                        logged server-side under ``request_id``.
    """
    session_id, session = request.app.state.sessions.get_or_create(body.session_id)

    async def _generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        # Hold the per-session lock for the whole stream so concurrent requests
        # on this session are serialised, not interleaved on one conversation.
        async with session.lock:
            try:
                async for event in session.agent.ask_stream(body.question):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception:
                # Never leak internal error detail (DB host, paths, creds) to
                # the client. Log the full exception server-side, correlated by
                # a request_id the client can quote when reporting the failure.
                request_id = uuid4().hex[:8]
                logger.exception(
                    "ask_stream failed (session=%s, request_id=%s)",
                    session_id,
                    request_id,
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "error",
                            "message": "internal error",
                            "request_id": request_id,
                        }
                    )
                    + "\n\n"
                )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
        },
    )


# ---------------------------------------------------------------------------
# Memory — training
# ---------------------------------------------------------------------------


@router.post("/memory/train", response_model=TrainResponse)
async def train(request: Request, body: TrainRequest) -> TrainResponse:
    """Add a verified training entry to the gold-standard collection."""
    memory = request.app.state.agent.agent_memory
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory layer is disabled")
    entry = TrainingEntry(
        id=make_entry_id(),
        question=body.question,
        tool_name=body.tool_name,
        tool_args=body.tool_args,
        result_summary=body.result_summary,
    )
    await memory.train(entry)
    return TrainResponse(imported=1)


@router.get("/memory/export", response_model=ExportResponse)
async def export_memory(request: Request) -> ExportResponse:
    """Export all memory entries (tool-usage, text, training) as JSON."""
    memory = request.app.state.agent.agent_memory
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory layer is disabled")
    entries = await memory.export_all()
    return ExportResponse(entries=entries, count=len(entries))


@router.post("/memory/import", response_model=TrainResponse)
async def import_memory(request: Request, body: list[dict]) -> TrainResponse:
    """Bulk-import entries in the export_all() format."""
    memory = request.app.state.agent.agent_memory
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory layer is disabled")
    imported = await memory.import_all(body)
    return TrainResponse(imported=imported)
