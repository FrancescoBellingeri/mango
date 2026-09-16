"""Mango FastAPI server.

Usage::

    from mango.servers.fastapi import MangoFastAPIServer

    server = MangoFastAPIServer(agent, user_for=my_auth)   # my_auth(request) -> user, or raise 401
    server.run()                                           # http://localhost:8000
    server.run(host="0.0.0.0", port=9000)

The underlying FastAPI app is accessible via server.app if you need to
mount it into an existing ASGI application.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mango.agent.agent import MangoAgent
from mango.servers.fastapi.routes import UserResolver, mango_router

logger = logging.getLogger(__name__)


def _parse_origins(value: str | None) -> list[str]:
    """Turn a comma-separated origin list into a clean list (default ``*``)."""
    if not value:
        return ["*"]
    origins = [o.strip() for o in value.split(",") if o.strip()]
    return origins or ["*"]


class MangoFastAPIServer:
    """Wraps a MangoAgent in a production-ready FastAPI/uvicorn server.

    Args:
        agent: A fully configured MangoAgent. setup() is called automatically
               on first use if it has not been called yet.
        user_for: Host callback ``(request) -> user`` (sync or async). This is
                  where *your* authentication lives: read your session cookie,
                  verify your JWT, check your API key — raise
                  ``fastapi.HTTPException(401)`` to reject, return your user
                  object to accept. That object is handed, untouched, to the
                  access-control middlewares (see :mod:`mango.middleware`).
                  Without it the server runs **open** (anonymous) and logs a
                  warning — fine on localhost, never on a network.
        cors_origins: Comma-separated list (or list) of allowed browser
                      origins. Defaults to ``MANGO_CORS_ORIGINS`` or ``*``.
                      With ``*`` credentials are disabled, as the CORS spec
                      requires; list explicit origins to allow cookies.
        session_ttl_seconds: Idle timeout after which a conversation is dropped.
        max_sessions: Cap on live conversations (least-recently-used evicted).
    """

    def __init__(
        self,
        agent: MangoAgent,
        *,
        user_for: UserResolver | None = None,
        cors_origins: str | list[str] | None = None,
        session_ttl_seconds: float = 1800.0,
        max_sessions: int = 1000,
    ) -> None:
        self._agent = agent
        self._user_for = user_for
        if isinstance(cors_origins, list):
            self._cors_origins = [o.strip() for o in cors_origins if o.strip()] or ["*"]
        else:
            self._cors_origins = _parse_origins(cors_origins or os.getenv("MANGO_CORS_ORIGINS"))
        self._session_ttl = session_ttl_seconds
        self._max_sessions = max_sessions
        self._app = self._build_app()

    @property
    def app(self) -> FastAPI:
        """The underlying FastAPI application instance."""
        return self._app

    def run(self, host: str = "0.0.0.0", port: int = 8000, **uvicorn_kwargs) -> None:
        """Start the uvicorn server (blocking).

        Args:
            host: Network interface to bind to. Default: '0.0.0.0'.
            port: TCP port. Default: 8000.
            **uvicorn_kwargs: Extra keyword arguments forwarded to uvicorn.run().
        """
        import uvicorn

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
            )
        uvicorn.run(self._app, host=host, port=port, **uvicorn_kwargs)

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="Mango API",
            description="Natural language interface for MongoDB",
            version="0.2.0",
        )

        allow_all = "*" in self._cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if allow_all else self._cors_origins,
            # The CORS spec forbids credentials with a wildcard origin.
            allow_credentials=not allow_all,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        router = mango_router(
            self._agent,
            user_for=self._user_for,
            session_ttl_seconds=self._session_ttl,
            max_sessions=self._max_sessions,
        )
        app.include_router(router, prefix="/api/v1")
        # Root agent exposed for introspection; per-client conversations are
        # isolated through the router's session manager.
        app.state.agent = self._agent
        app.state.sessions = router.sessions  # type: ignore[attr-defined]

        if self._user_for is None:
            logger.warning(
                "No user_for configured: the Mango API is OPEN — every request "
                "is anonymous (queries, memory training/import/export). Pass "
                "user_for=<your auth callback> before exposing it on a network."
            )
        if allow_all:
            logger.warning(
                "CORS allows any origin ('*'). Set MANGO_CORS_ORIGINS to your "
                "frontend origin(s) in production."
            )

        return app
