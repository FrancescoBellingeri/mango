"""Access-control middleware: the host decides, Mango enforces.

Every test drives the real agent loop with a scripted LLM and an opaque
``user`` object; the middlewares read that object and must block, rewrite or
mask exactly what they promise — through every path a value can take to the
LLM (tool results, auto-schema, prompt schema, value hints).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI

from mango.agent.agent import MangoAgent
from mango.core.access import TurnContext
from mango.core.types import AccessDenied, QueryRequest
from mango.llm.models import LLMResponse, ToolCall
from mango.middleware import (
    AuditLog,
    Budget,
    CollectionAccess,
    DenyTools,
    RedactFields,
    RowFilter,
)
from mango.servers.fastapi import MangoFastAPIServer, mango_router


@dataclass
class User:
    id: str
    role: str = "analyst"
    product: str = "Widget"


ADMIN = User(id="root", role="admin")
ANALYST = User(id="ana", role="analyst")

ROLE_COLLECTIONS = {"analyst": ["users"], "admin": "*"}


def _tc(name: str, **args) -> ToolCall:
    return ToolCall(tool_call_id=f"call_{name}", tool_name=name, tool_args=args)


def _agent(MockLLM, backend, registry, responses, *, introspect=False) -> tuple[MangoAgent, object]:
    llm = MockLLM(responses)
    agent = MangoAgent(llm_service=llm, tool_registry=registry, db=backend, introspect=introspect)
    agent.setup()
    return agent, llm


def _preview(events: list[dict], tool: str) -> str:
    for e in events:
        if e["type"] == "tool_result" and e["tool_name"] == tool:
            return e["preview"]
    raise AssertionError(f"no tool_result for {tool}: {events}")


async def _stream(agent: MangoAgent, question: str, user) -> list[dict]:
    return [e async for e in agent.ask_stream(question, user=user)]


# ---------------------------------------------------------------------------
# CollectionAccess
# ---------------------------------------------------------------------------


class TestCollectionAccess:
    async def test_denied_collection_is_blocked_and_reported(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[_tc("run_mql", operation="count", collection="orders")]),
            LLMResponse(text="You cannot see orders.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(CollectionAccess(lambda u: ROLE_COLLECTIONS[u.role]))

        events = await _stream(agent, "how many orders?", ANALYST)
        preview = _preview(events, "run_mql")
        assert "ACCESS DENIED" in preview and "orders" in preview
        assert events[-2]["text"] == "You cannot see orders."

    async def test_admin_passes(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[_tc("run_mql", operation="count", collection="orders")]),
            LLMResponse(text="3 orders.", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(CollectionAccess(lambda u: ROLE_COLLECTIONS[u.role]))
        events = await _stream(agent, "how many orders?", ADMIN)
        assert '"count":3' in _preview(events, "run_mql")

    async def test_listing_and_lookup_are_scoped_by_the_runner(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[
                _tc("list_collections"),
                _tc("run_mql", operation="aggregate", collection="users", pipeline=[
                    {"$lookup": {"from": "orders", "localField": "_id", "foreignField": "user_id", "as": "o"}},
                ]),
            ]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(CollectionAccess(lambda u: ROLE_COLLECTIONS[u.role]))
        events = await _stream(agent, "join", ANALYST)
        listing = _preview(events, "list_collections")
        assert "users" in listing and "orders" not in listing
        assert "'orders' is not accessible" in _preview(events, "run_mql")

    async def test_hidden_collection_never_reaches_the_prompt(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="ok", tool_calls=[]), LLMResponse(text="ok", tool_calls=[])]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses, introspect=True)
        assert "orders" in agent._schema
        agent.use(CollectionAccess(lambda u: ROLE_COLLECTIONS[u.role]))

        await agent.ask("list products in orders", user=ANALYST)
        prompt = llm.calls[0]["system_prompt"]
        assert "users" in prompt
        assert "orders" not in prompt
        assert "Widget" not in prompt  # no value hints from the hidden collection either

        await agent.ask("list products in orders", user=ADMIN)
        assert "orders" in llm.calls[1]["system_prompt"]


# ---------------------------------------------------------------------------
# RowFilter
# ---------------------------------------------------------------------------


class TestRowFilter:
    async def test_find_count_and_aggregate_are_filtered(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[
                _tc("run_mql", operation="find", collection="orders", projection={"_id": 0, "product": 1}),
                _tc("run_mql", operation="count", collection="orders"),
                _tc("run_mql", operation="aggregate", collection="orders",
                    pipeline=[{"$group": {"_id": "$product", "n": {"$sum": 1}}}]),
            ]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(RowFilter("orders", lambda u: {} if u.role == "admin" else {"product": u.product}))

        events = await _stream(agent, "orders", ANALYST)
        previews = [e["preview"] for e in events if e["type"] == "tool_result"]
        assert previews[0].count("Widget") == 2 and "Gadget" not in previews[0]
        assert '"count":2' in previews[1]
        assert "Gadget" not in previews[2] and "Widget" in previews[2]

    def test_lookup_forms_are_rewritten_and_graphlookup_denied(self):
        mw = RowFilter("orders", lambda u: {"region": u.product})
        ctx = TurnContext(user=User(id="x", product="EU"))

        simple = QueryRequest(operation="aggregate", collection="users", pipeline=[
            {"$lookup": {"from": "orders", "localField": "_id", "foreignField": "user_id", "as": "o"}},
        ])
        out = mw.before_query(ctx, simple).pipeline[0]["$lookup"]
        assert "localField" not in out and out["let"] == {"mango_local": "$_id"}
        assert out["pipeline"][0] == {"$match": {"region": "EU"}}
        assert out["pipeline"][1] == {"$match": {"$expr": {"$eq": ["$user_id", "$$mango_local"]}}}

        piped = QueryRequest(operation="aggregate", collection="users", pipeline=[
            {"$lookup": {"from": "orders", "pipeline": [{"$limit": 1}], "as": "o"}},
            {"$unionWith": "orders"},
        ])
        out_p = mw.before_query(ctx, piped).pipeline
        assert out_p[0]["$lookup"]["pipeline"][0] == {"$match": {"region": "EU"}}
        assert out_p[1]["$unionWith"]["pipeline"][0] == {"$match": {"region": "EU"}}

        graph = QueryRequest(operation="aggregate", collection="users", pipeline=[
            {"$graphLookup": {"from": "orders", "startWith": "$x", "connectFromField": "a",
                              "connectToField": "b", "as": "g"}},
        ])
        with pytest.raises(AccessDenied):
            mw.before_query(ctx, graph)

        untouched = QueryRequest(operation="find", collection="users", filter={"x": 1})
        assert mw.before_query(ctx, untouched) is untouched


# ---------------------------------------------------------------------------
# RedactFields
# ---------------------------------------------------------------------------


class TestRedactFields:
    @pytest.fixture
    def redact(self):
        return RedactFields(lambda u: [] if u.role == "admin" else ["users.age"])

    async def test_rows_and_describe_are_masked(self, MockLLM, mongo_backend, tool_registry, redact):
        responses = [
            LLMResponse(text="", tool_calls=[
                _tc("run_mql", operation="find", collection="users"),
                _tc("describe_collection", collection="users"),
            ]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(redact)
        events = await _stream(agent, "users", ANALYST)
        rows = _preview(events, "run_mql")
        assert "[REDACTED]" in rows and '"age":30' not in rows and "Alice" in rows
        desc = _preview(events, "describe_collection")
        assert '"age":30' not in desc and '"redacted":true' in desc

    async def test_queries_referencing_the_field_are_denied(self, MockLLM, mongo_backend, tool_registry, redact):
        calls = [
            _tc("run_mql", operation="find", collection="users", filter={"age": {"$gt": 1}}),
            _tc("run_mql", operation="find", collection="users", projection={"age": 1}),
            _tc("run_mql", operation="aggregate", collection="users",
                pipeline=[{"$group": {"_id": "$age"}}]),
            _tc("run_mql", operation="distinct", collection="users", distinct_field="age"),
            _tc("inspect_field", collection="users", field="age"),
        ]
        responses = [LLMResponse(text="", tool_calls=calls), LLMResponse(text="done", tool_calls=[])]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(redact)
        events = await _stream(agent, "ages", ANALYST)
        previews = [e["preview"] for e in events if e["type"] == "tool_result"]
        assert len(previews) == 5
        assert all("ACCESS DENIED" in p for p in previews), previews

    async def test_admin_is_unaffected(self, MockLLM, mongo_backend, tool_registry, redact):
        responses = [
            LLMResponse(text="", tool_calls=[_tc("run_mql", operation="find", collection="users", filter={"age": {"$gt": 1}})]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(redact)
        events = await _stream(agent, "ages", ADMIN)
        assert '"age":30' in _preview(events, "run_mql")

    async def test_prompt_schema_loses_sample_values(self, MockLLM, mongo_backend, tool_registry):
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, [LLMResponse(text="ok", tool_calls=[])], introspect=True)
        agent.use(RedactFields(lambda u: ["users.name"]))
        await agent.ask("who is alice", user=ANALYST)
        assert "Alice" not in llm.calls[0]["system_prompt"]
        # The shared schema object is untouched.
        name_field = next(f for f in agent._schema["users"].fields if f.path == "name")
        assert name_field.sample_values


# ---------------------------------------------------------------------------
# DenyTools / Budget / AuditLog / decorators
# ---------------------------------------------------------------------------


class TestDenyTools:
    async def test_tool_hidden_from_llm_and_refused(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[_tc("inspect_field", collection="users", field="name")]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, llm = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(DenyTools(lambda u: [] if u.role == "admin" else ["inspect_field"]))
        events = await _stream(agent, "q", ANALYST)
        offered = [t.name for t in llm.calls[0]["tools"]]
        assert "inspect_field" not in offered and "run_mql" in offered
        assert "ACCESS DENIED" in _preview(events, "inspect_field")


class TestBudget:
    async def test_turn_budget(self, MockLLM, mongo_backend, tool_registry):
        responses = [LLMResponse(text="a", tool_calls=[]), LLMResponse(text="b", tool_calls=[])]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(Budget(max_turns_per_day=1))
        await agent.ask("one", user=ANALYST)
        with pytest.raises(AccessDenied, match="turn budget"):
            await agent.ask("two", user=ANALYST)
        await agent.ask("other user is fine", user=ADMIN)

    async def test_query_budget(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[
                _tc("run_mql", operation="count", collection="users"),
                _tc("run_mql", operation="count", collection="orders"),
            ]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(Budget(max_queries_per_day=1))
        events = await _stream(agent, "q", ANALYST)
        previews = [e["preview"] for e in events if e["type"] == "tool_result"]
        assert '"count":3' in previews[0]
        assert "ACCESS DENIED" in previews[1] and "query budget" in previews[1]


class TestAuditLog:
    async def test_events_are_emitted(self, MockLLM, mongo_backend, tool_registry):
        sink: list[dict] = []
        responses = [
            LLMResponse(text="", tool_calls=[
                _tc("run_mql", operation="count", collection="users"),
                _tc("run_mql", operation="count", collection="orders"),
            ]),
            LLMResponse(text="done", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)
        agent.use(CollectionAccess(lambda u: ROLE_COLLECTIONS[u.role]))
        agent.use(AuditLog(sink=sink.append))
        await agent.ask("q", user=ANALYST)
        kinds = [e["event"] for e in sink]
        assert kinds == ["query", "denied", "turn"]
        assert sink[0]["user"] == "ana" and sink[0]["collection"] == "users" and sink[0]["row_count"] == 1
        assert sink[1]["tool"] == "run_mql"
        assert sink[2]["denied"] == 1 and sink[2]["tool_calls"] == ["run_mql", "run_mql"]
        assert all(e["turn_id"] == sink[0]["turn_id"] for e in sink)


class TestDecorators:
    async def test_before_query_and_before_answer(self, MockLLM, mongo_backend, tool_registry):
        responses = [
            LLMResponse(text="", tool_calls=[_tc("run_mql", operation="count", collection="orders")]),
            LLMResponse(text="secret answer", tool_calls=[]),
        ]
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, responses)

        @agent.before_query
        def interns_cannot_see_orders(ctx, query):
            if ctx.user.role == "intern" and query.collection == "orders":
                raise AccessDenied("orders are off-limits for interns")
            return query

        @agent.before_answer
        def stamp(ctx, text):
            return f"[{ctx.user.id}] {text}"

        events = await _stream(agent, "q", User(id="tim", role="intern"))
        assert "off-limits" in _preview(events, "run_mql")
        assert events[-2]["text"] == "[tim] secret answer"
        assert agent.conversation_length and agent._conversation[-1].content == "[tim] secret answer"

    def test_use_rejects_bare_functions(self, MockLLM, mongo_backend, tool_registry):
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, [])
        with pytest.raises(TypeError):
            agent.use(lambda ctx, q: q)

    async def test_new_session_shares_the_chain(self, MockLLM, mongo_backend, tool_registry):
        agent, _ = _agent(MockLLM, mongo_backend, tool_registry, [LLMResponse(text="x", tool_calls=[])])
        child = agent.new_session()
        agent.use(Budget(max_turns_per_day=1))
        await child.ask("one", user=ANALYST)
        with pytest.raises(AccessDenied):
            await child.ask("two", user=ANALYST)


# ---------------------------------------------------------------------------
# Router: user_for → sessions bound to users, memory endpoints gated
# ---------------------------------------------------------------------------


class RouterStub:
    def __init__(self) -> None:
        self.agent_memory = None
        self.seen: list = []
        self.deny_for = "bob"

    def new_session(self):
        child = RouterStub()
        child.seen = self.seen
        child.deny_for = self.deny_for
        return child

    def check_tool_access(self, user, tool_name, args=None):
        if user and user.get("id") == self.deny_for:
            raise AccessDenied(f"{tool_name} not allowed for {user['id']}")

    async def ask_stream(self, question, user=None):
        self.seen.append(user)
        if user and user.get("id") == "blocked":
            raise AccessDenied("daily budget exhausted")
        yield {"type": "answer", "text": f"{user['id'] if user else 'anon'}:{question}"}


def _app(stub) -> FastAPI:
    app = FastAPI()
    app.include_router(
        mango_router(stub, user_for=lambda r: {"id": r.headers.get("x-user", "anon")}),
        prefix="/mango",
    )
    return app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


class TestRouter:
    async def test_user_reaches_the_agent_and_sessions_are_bound(self):
        stub = RouterStub()
        async with _client(_app(stub)) as c:
            r = await c.post("/mango/ask/stream", json={"question": "hi", "session_id": "s1"},
                             headers={"x-user": "alice"})
            assert r.status_code == 200 and "alice:hi" in r.text
            assert stub.seen == [{"id": "alice"}]

            hijack = await c.post("/mango/ask/stream", json={"question": "hi", "session_id": "s1"},
                                  headers={"x-user": "bob"})
            assert hijack.status_code == 403

            again = await c.post("/mango/ask/stream", json={"question": "again", "session_id": "s1"},
                                 headers={"x-user": "alice"})
            assert again.status_code == 200

    async def test_before_turn_denial_is_a_safe_error_event(self):
        async with _client(_app(RouterStub())) as c:
            r = await c.post("/mango/ask/stream", json={"question": "hi"}, headers={"x-user": "blocked"})
        assert r.status_code == 200
        assert '"code": "access_denied"' in r.text and "daily budget exhausted" in r.text

    async def test_memory_endpoints_are_gated_through_the_agent(self):
        stub = RouterStub()

        class Mem:
            async def export_all(self):
                return []

        stub.agent_memory = Mem()
        async with _client(_app(stub)) as c:
            ok = await c.get("/mango/memory/export", headers={"x-user": "alice"})
            no = await c.get("/mango/memory/export", headers={"x-user": "bob"})
        assert ok.status_code == 200
        assert no.status_code == 403 and "memory_export not allowed" in no.text

    async def test_server_accepts_user_for(self):
        stub = RouterStub()
        app = MangoFastAPIServer(stub, user_for=lambda r: {"id": r.headers.get("x-user")}).app
        async with _client(app) as c:
            r = await c.post("/api/v1/ask/stream", json={"question": "q"}, headers={"x-user": "carol"})
        assert "carol:q" in r.text
