"""Access-control middleware: the host decides, Mango enforces.

Mango does not know what a "user" or a "role" is. The host application passes
an opaque ``user`` object to :meth:`MangoAgent.ask`; middlewares registered on
the agent read that object and decide, at fixed interception points, whether
a tool call / query / result / answer may proceed, must be rewritten, or must
be blocked (by raising :class:`~mango.core.types.AccessDenied`).

Interception points (all optional, all no-ops by default)::

    before_turn(ctx)                        budgets, blanket denials
    filter_tools(ctx, tool_defs)            hide tools from the LLM
    before_tool(ctx, tool_name, args)       block / rewrite any tool call
    before_query(ctx, query)                block / rewrite a run_mql QueryRequest
    after_query(ctx, query, rows)           mask / filter the rows
    after_tool(ctx, tool_name, args, result) mask any tool result (schema, samples)
    before_answer(ctx, text)                last check on the final text
    after_turn(ctx, stats)                  audit
    filter_schema(ctx, schema)              hide collections/fields from the prompt

Middlewares that need to restrict *which collections* a query may touch
implement :meth:`Middleware.collection_policy`; the resulting policy is
applied by the backend runner itself (listing, introspection and execution,
including ``$lookup`` / ``$unionWith`` targets), so it cannot be bypassed by
a pipeline the LLM writes.

The active turn (context + chain) is published in a :mod:`contextvars`
variable so that tools and the runner can consult it without threading it
through every signature. ``asyncio.to_thread`` copies the context, so this
works across the sync/async boundary too.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from mango.core.security import CollectionPolicy
from mango.core.types import AccessDenied, QueryRequest

__all__ = [
    "AccessDenied",
    "TurnContext",
    "Middleware",
    "FunctionMiddleware",
    "MiddlewareChain",
    "TurnAccess",
    "current_access",
    "current_collection_policy",
    "apply_before_query",
    "apply_after_query",
    "user_key",
]


def user_key(user: Any) -> str:
    """Best-effort stable identifier for an opaque user object.

    Looks for ``id`` / ``sub`` / ``email`` / ``username`` as attributes or
    dict keys; falls back to ``str(user)``. Used to bind sessions to users
    and to key budgets and audit records.
    """
    if user is None:
        return "anonymous"
    for attr in ("id", "sub", "email", "username"):
        val = getattr(user, attr, None)
        if val is None and isinstance(user, dict):
            val = user.get(attr)
        if val is not None:
            return str(val)
    return str(user)[:64]


@dataclass
class TurnContext:
    """What a middleware knows about the call being evaluated."""

    user: Any = None
    question: str = ""
    session_id: str | None = None
    turn_id: str = field(default_factory=lambda: uuid4().hex[:12])
    # Free-form scratch space shared by middlewares within one turn.
    state: dict[str, Any] = field(default_factory=dict)


class Middleware:
    """Base class: override only the hooks you need.

    Every hook receives the :class:`TurnContext` first. Hooks that return a
    value must return the (possibly rewritten) input; raising
    :class:`AccessDenied` blocks the call with a message the LLM will relay.
    """

    def before_turn(self, ctx: TurnContext) -> None:
        return None

    def filter_tools(self, ctx: TurnContext, tool_defs: list) -> list:
        return tool_defs

    def before_tool(self, ctx: TurnContext, tool_name: str, args: dict) -> dict:
        return args

    def before_query(self, ctx: TurnContext, query: QueryRequest) -> QueryRequest:
        return query

    def after_query(
        self, ctx: TurnContext, query: QueryRequest, rows: list[dict]
    ) -> list[dict]:
        return rows

    def after_tool(self, ctx: TurnContext, tool_name: str, args: dict, result):
        return result

    def before_answer(self, ctx: TurnContext, text: str) -> str:
        return text

    def after_turn(self, ctx: TurnContext, stats: dict) -> None:
        return None

    def collection_policy(self, ctx: TurnContext) -> CollectionPolicy | None:
        """Return a per-call collection policy, or None for no restriction."""
        return None

    def filter_schema(self, ctx: TurnContext, schema: dict) -> dict:
        """Rewrite the introspected schema before it is shown to the LLM."""
        return schema


class FunctionMiddleware(Middleware):
    """Wrap a plain function as a single-hook middleware (used by decorators)."""

    def __init__(self, hook: str, fn: Callable) -> None:
        if not hasattr(Middleware, hook) or hook.startswith("_"):
            raise ValueError(f"Unknown middleware hook '{hook}'.")
        self._hook = hook
        self._fn = fn
        setattr(self, hook, fn)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FunctionMiddleware({self._hook}={getattr(self._fn, '__name__', self._fn)!r})"


class MiddlewareChain:
    """Ordered list of middlewares; runs each hook in registration order."""

    def __init__(self, middlewares: Iterable[Middleware] | None = None) -> None:
        self._items: list[Middleware] = list(middlewares or [])

    def add(self, middleware: Middleware) -> None:
        if not isinstance(middleware, Middleware):
            raise TypeError(
                "agent.use() expects a Middleware instance; to register a bare "
                "function use the @agent.before_query / @agent.before_tool / "
                "@agent.after_query / @agent.before_answer decorators."
            )
        self._items.append(middleware)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    # -- hooks ---------------------------------------------------------

    def before_turn(self, ctx: TurnContext) -> None:
        for m in self._items:
            m.before_turn(ctx)

    def filter_tools(self, ctx: TurnContext, tool_defs: list) -> list:
        for m in self._items:
            tool_defs = m.filter_tools(ctx, tool_defs)
        return tool_defs

    def before_tool(self, ctx: TurnContext, tool_name: str, args: dict) -> dict:
        for m in self._items:
            args = m.before_tool(ctx, tool_name, args)
        return args

    def before_query(self, ctx: TurnContext, query: QueryRequest) -> QueryRequest:
        for m in self._items:
            query = m.before_query(ctx, query)
        return query

    def after_query(
        self, ctx: TurnContext, query: QueryRequest, rows: list[dict]
    ) -> list[dict]:
        for m in self._items:
            rows = m.after_query(ctx, query, rows)
        return rows

    def after_tool(self, ctx: TurnContext, tool_name: str, args: dict, result):
        for m in self._items:
            result = m.after_tool(ctx, tool_name, args, result)
        return result

    def before_answer(self, ctx: TurnContext, text: str) -> str:
        for m in self._items:
            text = m.before_answer(ctx, text)
        return text

    def after_turn(self, ctx: TurnContext, stats: dict) -> None:
        for m in self._items:
            m.after_turn(ctx, stats)

    def filter_schema(self, ctx: TurnContext, schema: dict) -> dict:
        for m in self._items:
            schema = m.filter_schema(ctx, schema)
        return schema

    def collection_policy(self, ctx: TurnContext) -> CollectionPolicy | None:
        """Combine every middleware's policy: a name must pass all of them."""
        combined: CollectionPolicy | None = None
        for m in self._items:
            policy = m.collection_policy(ctx)
            if policy is None:
                continue
            combined = policy if combined is None else combined.combined(policy)
        return combined


# ---------------------------------------------------------------------------
# Active-turn registry (contextvar)
# ---------------------------------------------------------------------------


@dataclass
class TurnAccess:
    """The chain + context currently governing tool execution."""

    ctx: TurnContext
    chain: MiddlewareChain
    collection_policy: CollectionPolicy | None = None


_current: contextvars.ContextVar[TurnAccess | None] = contextvars.ContextVar(
    "mango_turn_access", default=None
)


def activate(access: TurnAccess) -> contextvars.Token:
    """Publish *access* for the current task; returns a token for :func:`deactivate`."""
    return _current.set(access)


def deactivate(token: contextvars.Token) -> None:
    _current.reset(token)


def current_access() -> TurnAccess | None:
    """The :class:`TurnAccess` of the running turn, or None outside a turn."""
    return _current.get()


def current_collection_policy() -> CollectionPolicy | None:
    """Per-call collection policy for the running turn (consulted by runners)."""
    access = _current.get()
    return access.collection_policy if access is not None else None


def apply_before_query(query: QueryRequest) -> QueryRequest:
    """Run the active chain's ``before_query`` hook (no-op outside a turn)."""
    access = _current.get()
    if access is None:
        return query
    return access.chain.before_query(access.ctx, query)


def apply_after_query(query: QueryRequest, rows: list[dict]) -> list[dict]:
    """Run the active chain's ``after_query`` hook (no-op outside a turn)."""
    access = _current.get()
    if access is None:
        return rows
    return access.chain.after_query(access.ctx, query, rows)
