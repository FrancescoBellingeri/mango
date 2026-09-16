"""FastAPI routes, built per agent by :func:`mango_router`.

Mount Mango into an existing app in three lines::

    from mango.servers.fastapi import mango_router

    app.include_router(
        mango_router(agent, user_for=lambda request: request.state.user),
        prefix="/mango",
    )

``user_for`` is the only contact point with the host's authentication: it
receives the request and returns the host's own user object (or awaits one).
That object is passed untouched to ``agent.ask_stream(question, user=...)``
where the access-control middlewares read it. Raise
``fastapi.HTTPException(status_code=401)`` inside ``user_for`` to reject a
request. When ``user_for`` is omitted, requests are anonymous.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from mango.core.types import AccessDenied
from mango.memory import make_entry_id
from mango.memory.models import TrainingEntry
from mango.servers.fastapi.models import (
    MAX_IMPORT_ENTRIES,
    AskRequest,
    ExportResponse,
    HealthResponse,
    TrainRequest,
    TrainResponse,
)
from mango.servers.fastapi.sessions import SessionManager

logger = logging.getLogger(__name__)

UserResolver = Callable[[Request], Any | Awaitable[Any]]

# Pseudo tool names the memory endpoints are gated with, so a DenyTools
# middleware (or any before_tool hook) can restrict them per user.
MEMORY_TRAIN = "memory_train"
MEMORY_IMPORT = "memory_import"
MEMORY_EXPORT = "memory_export"


def mango_router(
    agent: Any,
    *,
    user_for: UserResolver | None = None,
    session_ttl_seconds: float = 1800.0,
    max_sessions: int = 1000,
    memory_endpoints: bool = True,
) -> APIRouter:
    """Build the Mango API router for *agent*.

    Args:
        agent: A configured :class:`~mango.MangoAgent` (root; sessions are
            minted from it).
        user_for: Host callback ``(request) -> user`` (sync or async). Its
            result is opaque to Mango and handed to the middlewares.
        session_ttl_seconds: Idle timeout for conversations.
        max_sessions: Cap on live conversations (LRU eviction).
        memory_endpoints: Expose ``/memory/train|import|export``. Each call
            is gated through the agent's ``before_tool`` hooks with the pseudo
            tool names ``memory_train`` / ``memory_import`` / ``memory_export``.
    """
    router = APIRouter()
    sessions = SessionManager(
        agent, ttl_seconds=session_ttl_seconds, max_sessions=max_sessions
    )
    router.sessions = sessions  # type: ignore[attr-defined]  (introspection/tests)

    async def _resolve_user(request: Request) -> Any:
        if user_for is None:
            return None
        user = user_for(request)
        if inspect.isawaitable(user):
            user = await user
        return user

    def _authorize(user: Any, action: str, args: dict | None = None) -> None:
        check = getattr(agent, "check_tool_access", None)
        if check is None:
            return
        try:
            check(user, action, args or {})
        except AccessDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def _memory():
        memory = getattr(agent, "agent_memory", None)
        if memory is None:
            raise HTTPException(status_code=503, detail="Memory layer is disabled")
        return memory

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy", service="mango")

    # ------------------------------------------------------------------
    # Ask (natural language)
    # ------------------------------------------------------------------

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
        - ``error``       — ``{message, code, request_id}``. ``code`` is
                            ``access_denied`` when a middleware blocked the
                            turn (message is safe to show); otherwise
                            ``internal`` with a generic message — the detail
                            is logged server-side under ``request_id``.
        """
        user = await _resolve_user(request)
        try:
            session_id, session = sessions.get_or_create(body.session_id, user=user)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        async def _generate() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            # Hold the per-session lock for the whole stream so concurrent
            # requests on this session are serialised, not interleaved.
            async with session.lock:
                try:
                    async for event in session.agent.ask_stream(body.question, user=user):
                        yield f"data: {json.dumps(event)}\n\n"
                except AccessDenied as exc:
                    yield "data: " + json.dumps({
                        "type": "error", "code": "access_denied", "message": str(exc),
                    }) + "\n\n"
                except Exception:
                    # Never leak internal error detail (DB host, paths, creds).
                    request_id = uuid4().hex[:8]
                    logger.exception(
                        "ask_stream failed (session=%s, request_id=%s)", session_id, request_id
                    )
                    yield "data: " + json.dumps({
                        "type": "error", "code": "internal",
                        "message": "internal error", "request_id": request_id,
                    }) + "\n\n"

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            },
        )

    if not memory_endpoints:
        return router

    # ------------------------------------------------------------------
    # Memory — training / export / import
    # ------------------------------------------------------------------

    @router.post("/memory/train", response_model=TrainResponse)
    async def train(request: Request, body: TrainRequest) -> TrainResponse:
        """Add a verified training entry to the gold-standard collection."""
        user = await _resolve_user(request)
        _authorize(user, MEMORY_TRAIN, body.model_dump())
        memory = _memory()
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
        user = await _resolve_user(request)
        _authorize(user, MEMORY_EXPORT)
        memory = _memory()
        entries = await memory.export_all()
        return ExportResponse(entries=entries, count=len(entries))

    @router.post("/memory/import", response_model=TrainResponse)
    async def import_memory(request: Request, body: list[dict]) -> TrainResponse:
        """Bulk-import entries in the export_all() format."""
        user = await _resolve_user(request)
        _authorize(user, MEMORY_IMPORT, {"count": len(body)})
        memory = _memory()
        if len(body) > MAX_IMPORT_ENTRIES:
            raise HTTPException(
                status_code=413,
                detail=f"Too many entries ({len(body)}); max {MAX_IMPORT_ENTRIES} per request.",
            )
        imported = await memory.import_all(body)
        return TrainResponse(imported=imported)

    return router
