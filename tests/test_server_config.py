"""CORS parsing and input caps on the FastAPI server."""

from __future__ import annotations

import httpx
import pytest

from mango.servers.fastapi.main import MangoFastAPIServer, _parse_origins
from mango.servers.fastapi.models import MAX_IMPORT_ENTRIES, MAX_QUESTION_CHARS


class StubAgent:
    def __init__(self) -> None:
        self.agent_memory = None

    def new_session(self) -> "StubAgent":
        return StubAgent()

    async def ask_stream(self, question: str, user=None):
        yield {"type": "answer", "text": f"echo:{question}"}


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.fixture
def open_app():
    return MangoFastAPIServer(StubAgent()).app


class TestCors:
    def test_parse_origins_splits_on_commas(self):
        assert _parse_origins("https://a.com, https://b.com,") == ["https://a.com", "https://b.com"]
        assert _parse_origins(None) == ["*"]
        assert _parse_origins("  ") == ["*"]

    def _cors_options(self, app) -> dict:
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                return mw.kwargs
        raise AssertionError("CORSMiddleware not installed")

    def test_wildcard_disables_credentials(self, open_app):
        opts = self._cors_options(open_app)
        assert opts["allow_origins"] == ["*"]
        assert opts["allow_credentials"] is False

    def test_explicit_origins_enable_credentials(self, monkeypatch):
        monkeypatch.setenv("MANGO_CORS_ORIGINS", "https://a.com,https://b.com")
        app = MangoFastAPIServer(StubAgent()).app
        opts = self._cors_options(app)
        assert opts["allow_origins"] == ["https://a.com", "https://b.com"]
        assert opts["allow_credentials"] is True

    def test_constructor_list_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MANGO_CORS_ORIGINS", "https://env.com")
        app = MangoFastAPIServer(StubAgent(), cors_origins=["https://ctor.com"]).app
        assert self._cors_options(app)["allow_origins"] == ["https://ctor.com"]


class TestInputCaps:
    async def test_question_too_long_is_422(self, open_app):
        async with _client(open_app) as c:
            r = await c.post(
                "/api/v1/ask/stream", json={"question": "x" * (MAX_QUESTION_CHARS + 1)}
            )
        assert r.status_code == 422

    async def test_empty_question_is_422(self, open_app):
        async with _client(open_app) as c:
            r = await c.post("/api/v1/ask/stream", json={"question": ""})
        assert r.status_code == 422

    async def test_malformed_session_id_is_422(self, open_app):
        async with _client(open_app) as c:
            r = await c.post(
                "/api/v1/ask/stream", json={"question": "hi", "session_id": "../etc"}
            )
        assert r.status_code == 422

    async def test_import_too_large_is_413(self, open_app):
        class Mem:
            async def import_all(self, entries):
                return len(entries)

        open_app.state.agent.agent_memory = Mem()
        async with _client(open_app) as c:
            ok = await c.post("/api/v1/memory/import", json=[{}] * 3)
            big = await c.post("/api/v1/memory/import", json=[{}] * (MAX_IMPORT_ENTRIES + 1))
        assert ok.status_code == 200
        assert big.status_code == 413
