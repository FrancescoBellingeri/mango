"""Per-session agent management for the FastAPI server.

The server must not share a single conversation across clients. Each
``session_id`` gets its own agent (minted cheaply via ``agent.new_session()``)
plus its own lock, so:

  - different sessions never see each other's conversation history;
  - concurrent requests on the *same* session are serialised, preventing
    interleaved appends that would corrupt the message history
    (orphan ``tool_use`` → provider 400);
  - a session is bound to the user that created it: presenting its id with a
    different user is refused (``PermissionError``), so a leaked id cannot be
    replayed with other permissions.

Sessions are evicted by TTL (idle timeout) and capped in count (LRU), so
abandoned sessions do not leak memory.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from mango.core.access import user_key


@dataclass
class Session:
    """One client conversation: its agent, a serialising lock, last activity."""

    agent: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_seen: float = field(default_factory=time.monotonic)
    user_key: str = "anonymous"


class SessionManager:
    """Maps ``session_id`` → :class:`Session`, backed by ``root.new_session()``.

    Args:
        root_agent: The configured agent; ``new_session()`` is used to mint a
            fresh, independent conversation per session (shares schema/prompt).
        ttl_seconds: Idle timeout after which a session is evicted.
        max_sessions: Hard cap on live sessions; least-recently-used are dropped.
    """

    def __init__(
        self,
        root_agent: Any,
        *,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 1000,
    ) -> None:
        self._root = root_agent
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()

    def get_or_create(
        self, session_id: str | None, user: Any = None
    ) -> tuple[str, Session]:
        """Resolve *session_id* to a live session, creating one if needed.

        Synchronous and await-free on purpose: under asyncio's single-threaded
        cooperative scheduling this runs atomically, so concurrent requests for
        the same new id cannot double-create or race the eviction bookkeeping.

        Args:
            session_id: Client-supplied id, or None for a new session.
            user: Opaque user object (from the host's ``user_for``); the
                session is bound to its :func:`user_key`.

        Returns:
            ``(resolved_session_id, session)``. A new id is generated when the
            caller passes ``None``.

        Raises:
            PermissionError: The session exists but belongs to another user.
        """
        now = time.monotonic()
        self._evict_expired(now)
        key = user_key(user)

        if session_id is not None and session_id in self._sessions:
            session = self._sessions[session_id]
            if session.user_key != key:
                raise PermissionError("Session belongs to another user.")
            session.last_seen = now
            self._sessions.move_to_end(session_id)
            return session_id, session

        resolved = session_id or uuid4().hex
        agent = self._root.new_session()
        try:
            agent._session_id = resolved  # surfaced to middlewares via ctx
        except AttributeError:  # pragma: no cover - exotic agent stubs
            pass
        session = Session(agent=agent, user_key=key)
        self._sessions[resolved] = session
        self._sessions.move_to_end(resolved)
        self._evict_lru()
        return resolved, session

    def __len__(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_expired(self, now: float) -> None:
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]

    def _evict_lru(self) -> None:
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)  # drop least-recently-used
