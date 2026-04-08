"""Mango FastAPI server.

Usage::

    from mango.servers.fastapi import MangoFastAPIServer

    server = MangoFastAPIServer(agent)
    server.run()                          # http://localhost:8000
    server.run(host="0.0.0.0", port=9000)

The underlying FastAPI app is accessible via server.app if you need to
mount it into an existing ASGI application.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mango.agent.agent import MangoAgent
from mango.servers.fastapi.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class MangoFastAPIServer:
    """Wraps a MangoAgent in a production-ready FastAPI/uvicorn server.

    Args:
        agent: A fully configured MangoAgent. setup() is called automatically
               on first startup if it has not been called yet.
        cors_origins: Comma-separated list of allowed CORS origins.
                      Defaults to the MANGO_CORS_ORIGINS env var or '*'.
    """

    def __init__(
        self,
        agent: MangoAgent
    ) -> None:
        self._agent = agent
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
        uvicorn.run(self._app, host=host, port=port, **uvicorn_kwargs)

    def _build_app(self) -> FastAPI:

        app = FastAPI(
            title="Mango API",
            description="Natural language interface for MongoDB",
            version="0.1.0"
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=[os.getenv("MANGO_CORS_ORIGINS") or "*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(router, prefix="/api/v1")
        app.state.agent = self._agent
        return app
