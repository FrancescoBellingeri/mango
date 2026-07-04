"""Session-isolation tests for the FastAPI server.

The server hands every client the same ``app.state.agent`` instance and never
uses the returned ``session_id`` to look one up. Consequences:

  - Cross-session leak: two different ``session_id`` values share one
    ``MangoAgent._conversation``, so one caller sees another's history/data.
  - Race: concurrent requests interleave appends on that single conversation,
    producing orphan ``tool_use`` messages (provider 400).

These tests assert the POST-FIX behaviour and are therefore RED until the fix
lands (per-session agent map + per-session lock). They test observable HTTP
behaviour, not the fix's internals — the only contract assumed is that the
server isolates conversations per ``session_id`` (minted via
``agent.new_session()``) and serialises concurrent requests on one session.

The agent is stubbed on purpose: the defect lives entirely in the server
wiring (routes.py / main.py), not in MangoAgent. The stub mirrors the two
pieces of MangoAgent's contract the fix relies on: ``new_session()`` returns a
fresh, independent agent, and ``ask_stream()`` mutates per-instance state.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from mango.servers.fastapi.main import MangoFastAPIServer


# ---------------------------------------------------------------------------
# Stub agent — mirrors the slice of MangoAgent's contract the server touches
# ---------------------------------------------------------------------------


class StubAgent:
    """Minimal duck-type of MangoAgent for the ``/ask/stream`` path.

    ``conversation`` is the analog of ``MangoAgent._conversation`` (per-instance
    state). ``ask_stream`` mimics the real append pattern: record 'start', yield
    control at an await point (where the real agent awaits the LLM/tool), then
    record 'end' — so concurrent calls on one agent interleave observably.
    """

    def __init__(self) -> None:
        self.conversation: list[str] = []
        self.agent_memory = None
        self.children: list[StubAgent] = []  # sessions minted from this agent

    def new_session(self) -> "StubAgent":
        child = StubAgent()
        self.children.append(child)
        return child

    async def ask_stream(self, question: str):
        self.conversation.append(f"start:{question}")
        await asyncio.sleep(0.02)  # control returns to the event loop here
        self.conversation.append(f"end:{question}")
        yield {"type": "answer", "text": " | ".join(self.conversation)}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def server():
    """Yield ``(root_stub, http_client)`` wired to the real FastAPI app."""
    root = StubAgent()
    app = MangoFastAPIServer(root).app  # real app; stub only as the agent
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        yield root, client


async def _ask(client: httpx.AsyncClient, question: str, session_id: str | None) -> str:
    """POST /ask/stream and return the final ``answer`` text from the SSE."""
    body: dict = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    resp = await client.post("/api/v1/ask/stream", json=body)
    assert resp.status_code == 200
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[len("data: "):])
            if evt.get("type") == "answer":
                return evt["text"]
    raise AssertionError(f"no answer event in stream:\n{resp.text}")


def _served_conversation(root: StubAgent) -> list[str]:
    """Return the conversation list that actually handled the traffic.

    After the fix the server mints a per-session agent via ``new_session()``;
    before the fix everything lands on ``root`` itself. Exactly one conversation
    should be non-empty when all requests share a single session.
    """
    served = [a.conversation for a in (root, *root.children) if a.conversation]
    assert len(served) == 1, f"expected one serving conversation, got {len(served)}"
    return served[0]


# ---------------------------------------------------------------------------
# A. Cross-session leak
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    async def test_distinct_sessions_do_not_leak(self, server):
        root, client = server

        alice = await _ask(client, "ALICE-SECRET-XYZ", session_id="alice")
        bob = await _ask(client, "bobs-question", session_id="bob")

        assert "ALICE-SECRET-XYZ" in alice  # sanity: alice sees her own turn
        assert "ALICE-SECRET-XYZ" not in bob  # bob must NOT see alice's history

    async def test_same_session_keeps_context(self, server):
        """Guardrail: the fix must not make every request a brand-new session.

        A follow-up on the same ``session_id`` must still see prior turns.
        """
        root, client = server

        await _ask(client, "first-turn", session_id="s")
        second = await _ask(client, "second-turn", session_id="s")

        assert "first-turn" in second


# ---------------------------------------------------------------------------
# B. Concurrent race on a single session
# ---------------------------------------------------------------------------


class TestSessionSerialization:
    async def test_concurrent_same_session_requests_are_serialized(self, server):
        root, client = server
        n = 4

        await asyncio.gather(
            *(_ask(client, str(i), session_id="shared") for i in range(n))
        )

        conv = _served_conversation(root)
        # Serialized => each 'start:i' is immediately followed by its 'end:i'.
        for i in range(n):
            si = conv.index(f"start:{i}")
            ei = conv.index(f"end:{i}")
            assert ei == si + 1, f"request {i} interleaved by another: {conv}"


# ---------------------------------------------------------------------------
# C. Internal errors must not leak detail to the client
# ---------------------------------------------------------------------------


class TestErrorLeak:
    async def test_internal_error_detail_is_not_leaked(self):
        secret = "mongodb://user:pass@internal-db.prod:27017"

        class BoomAgent:
            agent_memory = None

            def new_session(self):
                return self

            async def ask_stream(self, question):
                raise RuntimeError(f"boom {secret}")
                yield  # unreachable — makes this an async generator

        app = MangoFastAPIServer(BoomAgent()).app
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post("/api/v1/ask/stream", json={"question": "hi"})

        assert resp.status_code == 200
        assert secret not in resp.text  # no infra detail reaches the client
        assert "boom" not in resp.text  # no raw exception text either

        events = [
            json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        err = next(e for e in events if e.get("type") == "error")
        assert err["message"] == "internal error"
        assert err["request_id"]  # correlation id for server-side lookup
