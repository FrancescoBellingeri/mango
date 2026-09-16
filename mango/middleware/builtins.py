"""Built-in middlewares. See :mod:`mango.core.access` for the hook contract."""

from __future__ import annotations

import copy
import json
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from mango.core.access import Middleware, TurnContext, user_key as _user_key
from mango.core.security import CollectionPolicy
from mango.core.types import AccessDenied, QueryRequest

REDACTED = "[REDACTED]"


# ---------------------------------------------------------------------------
# CollectionAccess
# ---------------------------------------------------------------------------


class CollectionAccess(Middleware):
    """Restrict which collections the user may see and query.

    ``collections_for(user)`` returns a list of collection names, or ``"*"``
    (or ``None``) for unrestricted access. Enforced by the runner on listing,
    introspection and execution — including ``$lookup``/``$unionWith``
    targets — so a pipeline cannot reach around it.
    """

    def __init__(self, collections_for: Callable[[Any], Iterable[str] | str | None]) -> None:
        self._for = collections_for

    def _policy(self, ctx: TurnContext) -> CollectionPolicy | None:
        cached = ctx.state.get("_collection_access")
        if cached is not None:
            return cached or None
        allowed = self._for(ctx.user)
        if allowed is None or allowed == "*":
            ctx.state["_collection_access"] = False
            return None
        policy = CollectionPolicy(allowed=list(allowed))
        ctx.state["_collection_access"] = policy
        return policy

    def collection_policy(self, ctx: TurnContext) -> CollectionPolicy | None:
        return self._policy(ctx)

    def filter_schema(self, ctx: TurnContext, schema: dict) -> dict:
        policy = self._policy(ctx)
        if policy is None:
            return schema
        return {name: info for name, info in schema.items() if policy.is_allowed(name)}

    def before_tool(self, ctx: TurnContext, tool_name: str, args: dict) -> dict:
        policy = self._policy(ctx)
        target = args.get("collection") if isinstance(args, dict) else None
        if policy is not None and isinstance(target, str) and not policy.is_allowed(target):
            raise AccessDenied(f"Collection '{target}' is not accessible to this user.")
        return args


# ---------------------------------------------------------------------------
# RowFilter
# ---------------------------------------------------------------------------


def _and(existing: dict | None, extra: dict) -> dict:
    if not existing:
        return copy.deepcopy(extra)
    return {"$and": [existing, copy.deepcopy(extra)]}


class RowFilter(Middleware):
    """Force a mandatory filter on one collection ("only your region").

    ``filter_for(user)`` returns an MQL filter document, or ``{}`` for no
    restriction. It is AND-ed into ``find``/``count``/``distinct`` filters,
    prepended as ``$match`` to aggregation pipelines, and pushed into
    ``$lookup``/``$unionWith`` stages that read the same collection (a
    ``localField``/``foreignField`` lookup is rewritten to the pipeline form).
    ``$graphLookup`` on the collection has nowhere to host a filter and is denied.
    """

    def __init__(self, collection: str, filter_for: Callable[[Any], dict | None]) -> None:
        self._collection = collection
        self._for = filter_for

    def before_query(self, ctx: TurnContext, query: QueryRequest) -> QueryRequest:
        extra = self._for(ctx.user) or {}
        if not extra:
            return query
        changes: dict[str, Any] = {}
        if query.pipeline:
            changes["pipeline"] = self._rewrite_pipeline(query.pipeline, extra, query.collection)
        elif query.collection == self._collection:
            changes["filter"] = _and(query.filter, extra)
        if query.pipeline and query.collection == self._collection:
            changes["pipeline"] = [{"$match": copy.deepcopy(extra)}] + changes["pipeline"]
        return replace(query, **changes) if changes else query

    def _rewrite_pipeline(self, pipeline: list, extra: dict, target: str) -> list:
        out: list = []
        for stage in pipeline:
            if not isinstance(stage, dict) or len(stage) != 1:
                out.append(stage)
                continue
            op, body = next(iter(stage.items()))
            if op == "$lookup" and isinstance(body, dict) and body.get("from") == self._collection:
                out.append({"$lookup": self._rewrite_lookup(body, extra)})
            elif op == "$graphLookup" and isinstance(body, dict) and body.get("from") == self._collection:
                raise AccessDenied(
                    f"$graphLookup on '{self._collection}' is not allowed for this user "
                    "(row-level filter cannot be applied)."
                )
            elif op == "$unionWith" and (
                body == self._collection
                or (isinstance(body, dict) and body.get("coll") == self._collection)
            ):
                sub = body.get("pipeline", []) if isinstance(body, dict) else []
                out.append({"$unionWith": {
                    "coll": self._collection,
                    "pipeline": [{"$match": copy.deepcopy(extra)}]
                    + self._rewrite_pipeline(sub, extra, self._collection),
                }})
            elif op == "$lookup" and isinstance(body, dict) and isinstance(body.get("pipeline"), list):
                new_body = dict(body)
                new_body["pipeline"] = self._rewrite_pipeline(body["pipeline"], extra, body.get("from", ""))
                out.append({"$lookup": new_body})
            elif op == "$facet" and isinstance(body, dict):
                out.append({"$facet": {
                    k: self._rewrite_pipeline(v, extra, target) if isinstance(v, list) else v
                    for k, v in body.items()
                }})
            else:
                out.append(stage)
        return out

    def _rewrite_lookup(self, body: dict, extra: dict) -> dict:
        new_body = dict(body)
        match = {"$match": copy.deepcopy(extra)}
        if isinstance(body.get("pipeline"), list):
            new_body["pipeline"] = [match] + self._rewrite_pipeline(body["pipeline"], extra, self._collection)
            return new_body
        local, foreign = body.get("localField"), body.get("foreignField")
        if not (isinstance(local, str) and isinstance(foreign, str)):
            raise AccessDenied(
                f"$lookup on '{self._collection}' must use localField/foreignField or a pipeline."
            )
        new_body.pop("localField", None)
        new_body.pop("foreignField", None)
        let = dict(body.get("let") or {})
        let["mango_local"] = f"${local}"
        new_body["let"] = let
        new_body["pipeline"] = [
            match,
            {"$match": {"$expr": {"$eq": [f"${foreign}", "$$mango_local"]}}},
        ]
        return new_body


# ---------------------------------------------------------------------------
# RedactFields
# ---------------------------------------------------------------------------


class RedactFields(Middleware):
    """Hide fields from the LLM entirely.

    ``fields_for(user)`` returns dotted ``"collection.field.path"`` entries.
    A query that references a hidden field (filter, projection, sort,
    ``$group`` key, ``$project`` expression…) is denied; ``inspect_field`` on
    it is denied; and any value that still reaches a tool result (query rows,
    sample documents, sample values) is replaced with ``[REDACTED]``.
    """

    def __init__(self, fields_for: Callable[[Any], Iterable[str]]) -> None:
        self._for = fields_for

    def _paths(self, ctx: TurnContext) -> dict[str, set[str]]:
        cached = ctx.state.get("_redact_paths")
        if cached is not None:
            return cached
        out: dict[str, set[str]] = {}
        for entry in self._for(ctx.user) or ():
            coll, _, path = entry.partition(".")
            if coll and path:
                out.setdefault(coll, set()).add(path)
        ctx.state["_redact_paths"] = out
        return out

    # -- hooks ---------------------------------------------------------

    def before_tool(self, ctx: TurnContext, tool_name: str, args: dict) -> dict:
        if tool_name == "inspect_field" and isinstance(args, dict):
            paths = self._paths(ctx).get(str(args.get("collection")), set())
            field = str(args.get("field", ""))
            if _path_hits(field, paths):
                raise AccessDenied(f"Field '{field}' is not accessible to this user.")
        return args

    def before_query(self, ctx: TurnContext, query: QueryRequest) -> QueryRequest:
        paths = self._paths(ctx).get(query.collection, set())
        if not paths:
            return query
        for doc in (query.filter, query.projection, query.sort, query.pipeline):
            hit = _find_field_reference(doc, paths)
            if hit:
                raise AccessDenied(
                    f"Field '{hit}' of '{query.collection}' is not accessible to this user."
                )
        if query.distinct_field and _path_hits(query.distinct_field, paths):
            raise AccessDenied(
                f"Field '{query.distinct_field}' of '{query.collection}' is not accessible to this user."
            )
        return query

    def filter_schema(self, ctx: TurnContext, schema: dict) -> dict:
        """Strip sample values / sample documents of hidden fields from the prompt schema."""
        redact = self._paths(ctx)
        if not redact:
            return schema
        out = dict(schema)
        for name, paths in redact.items():
            info = out.get(name)
            if info is None:
                continue
            info = copy.deepcopy(info)
            _scrub_fields(getattr(info, "fields", None) or [], paths)
            if getattr(info, "sample_documents", None):
                info.sample_documents = [_mask_doc(d, paths) for d in info.sample_documents]
            out[name] = info
        return out

    def after_tool(self, ctx: TurnContext, tool_name: str, args: dict, result):
        data = getattr(result, "data", None)
        collection = args.get("collection") if isinstance(args, dict) else None
        paths = self._paths(ctx).get(str(collection), set()) if collection else set()
        if not paths or not isinstance(data, dict):
            return result
        data = copy.deepcopy(data)
        for key in ("rows", "sample_documents"):
            if isinstance(data.get(key), list):
                data[key] = [_mask_doc(d, paths) for d in data[key]]
        if isinstance(data.get("fields"), list):
            for f in data["fields"]:
                if isinstance(f, dict) and _path_hits(str(f.get("path", "")), paths):
                    f["sample_values"] = None
                    f["redacted"] = True
        return replace(result, data=data)


def _scrub_fields(fields: list, redacted: set[str]) -> None:
    for f in fields:
        if _path_hits(str(getattr(f, "path", "")), redacted):
            f.sample_values = None
        sub = getattr(f, "sub_fields", None)
        if sub:
            _scrub_fields(sub, redacted)


def _path_hits(path: str, redacted: set[str]) -> bool:
    return any(path == p or path.startswith(p + ".") for p in redacted)


def _find_field_reference(obj: Any, redacted: set[str]) -> str | None:
    """Return the first redacted path referenced as a key or ``$field`` value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and not k.startswith("$") and _path_hits(k, redacted):
                return k
            hit = _find_field_reference(v, redacted)
            if hit:
                return hit
    elif isinstance(obj, list):
        for item in obj:
            hit = _find_field_reference(item, redacted)
            if hit:
                return hit
    elif isinstance(obj, str) and obj.startswith("$") and not obj.startswith("$$"):
        if _path_hits(obj[1:], redacted):
            return obj[1:]
    return None


def _mask_doc(doc: Any, redacted: set[str], prefix: str = "") -> Any:
    if isinstance(doc, list):
        return [_mask_doc(d, redacted, prefix) for d in doc]
    if not isinstance(doc, dict):
        return doc
    out: dict = {}
    for k, v in doc.items():
        full = f"{prefix}{k}"
        if _path_hits(full, redacted):
            out[k] = REDACTED
        elif isinstance(v, (dict, list)):
            out[k] = _mask_doc(v, redacted, full + ".")
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# DenyTools
# ---------------------------------------------------------------------------


class DenyTools(Middleware):
    """Remove tools for a user: hidden from the LLM *and* refused if called."""

    def __init__(self, tools_for: Callable[[Any], Iterable[str]]) -> None:
        self._for = tools_for

    def _denied(self, ctx: TurnContext) -> set[str]:
        cached = ctx.state.get("_denied_tools")
        if cached is None:
            cached = set(self._for(ctx.user) or ())
            ctx.state["_denied_tools"] = cached
        return cached

    def filter_tools(self, ctx: TurnContext, tool_defs: list) -> list:
        denied = self._denied(ctx)
        return [t for t in tool_defs if getattr(t, "name", None) not in denied]

    def before_tool(self, ctx: TurnContext, tool_name: str, args: dict) -> dict:
        if tool_name in self._denied(ctx):
            raise AccessDenied(f"Tool '{tool_name}' is not available to this user.")
        return args


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class Budget(Middleware):
    """Per-user daily caps on turns and executed queries.

    Counters live in process memory and reset at UTC midnight — good for a
    single server; behind several workers put the cap in your gateway instead.
    ``key_for(user)`` identifies the user (defaults to ``id``/``sub``/``email``).
    """

    def __init__(
        self,
        *,
        max_turns_per_day: int | None = None,
        max_queries_per_day: int | None = None,
        key_for: Callable[[Any], str] | None = None,
    ) -> None:
        self._max_turns = max_turns_per_day
        self._max_queries = max_queries_per_day
        self._key = key_for or _user_key
        self._counts: dict[tuple[str, str, str], int] = {}
        self._lock = threading.Lock()

    def _bump(self, ctx: TurnContext, kind: str, cap: int | None) -> None:
        if cap is None:
            return
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = (self._key(ctx.user), day, kind)
        with self._lock:
            n = self._counts.get(key, 0) + 1
            if n > cap:
                raise AccessDenied(
                    f"Daily {kind} budget of {cap} exhausted for this user."
                )
            self._counts[key] = n

    def before_turn(self, ctx: TurnContext) -> None:
        self._bump(ctx, "turn", self._max_turns)

    def before_query(self, ctx: TurnContext, query: QueryRequest) -> QueryRequest:
        self._bump(ctx, "query", self._max_queries)
        return query


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class AuditLog(Middleware):
    """Record who ran what: one event per executed query, denial and turn.

    Events are dicts; by default they are logged as JSON on the
    ``mango.audit`` logger, or handed to ``sink(event)`` when given.
    """

    def __init__(
        self,
        sink: Callable[[dict], None] | None = None,
        *,
        key_for: Callable[[Any], str] | None = None,
        logger_name: str = "mango.audit",
        max_query_chars: int = 500,
    ) -> None:
        self._sink = sink
        self._key = key_for or _user_key
        self._logger = logging.getLogger(logger_name)
        self._max_query_chars = max_query_chars

    def _emit(self, ctx: TurnContext, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user": self._key(ctx.user),
            "session_id": ctx.session_id,
            "turn_id": ctx.turn_id,
            **fields,
        }
        if self._sink is not None:
            self._sink(record)
        else:
            self._logger.info(json.dumps(record, default=str, ensure_ascii=False))

    def after_query(self, ctx: TurnContext, query: QueryRequest, rows: list[dict]) -> list[dict]:
        body = {
            "filter": query.filter, "pipeline": query.pipeline,
            "projection": query.projection, "sort": query.sort,
            "limit": query.limit, "distinct_field": query.distinct_field,
        }
        summary = json.dumps({k: v for k, v in body.items() if v}, default=str)
        self._emit(
            ctx, "query",
            collection=query.collection, operation=query.operation,
            query=summary[: self._max_query_chars], row_count=len(rows),
        )
        return rows

    def after_tool(self, ctx: TurnContext, tool_name: str, args: dict, result):
        if getattr(result, "error_kind", None) == "AccessDenied":
            self._emit(ctx, "denied", tool=tool_name, reason=getattr(result, "error", ""))
        return result

    def after_turn(self, ctx: TurnContext, stats: dict) -> None:
        self._emit(ctx, "turn", question=ctx.question[:200], **stats)
