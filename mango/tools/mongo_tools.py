"""Concrete MongoDB tools exposed to the LLM.

Each tool wraps a MongoRunner operation and exposes it via the Tool ABC.
The agent registers these tools in the ToolRegistry at startup.

Tools defined here:
    - list_collections   : list / group all collections in the database
    - search_collections : find collections by glob pattern
    - describe_collection: schema + indexes + sample docs for one collection
    - collection_stats   : document counts for all collections, sorted descending
    - run_mql            : execute a MQL query (find / aggregate / count / distinct)
    - explain_query      : explain a query step-by-step with execution stats
"""

from __future__ import annotations

import fnmatch
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

# Regex that matches ISO 8601 date strings with optional time/ms/tz parts.
_ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"           # date
    r"(?:[T ]\d{2}:\d{2}:\d{2}"     # optional time
    r"(?:\.\d+)?"                    # optional fractional seconds
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"   # optional timezone
)

# Regex that detects UUID segments inside collection names.
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

from mango.integrations import MongoRunner
from mango.core.types import QueryRequest
from mango.llm import ToolDef, ToolParam
from mango.tools.base import Tool, ToolResult
from mango.tools.validator import MQLValidator


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


class ListCollectionsTool(Tool):
    """Return all collection names, grouped by prefix when the DB is large."""

    _FLAT_THRESHOLD = 50
    _GROUP_MIN = 3

    def __init__(self, backend: MongoRunner) -> None:
        self._backend = backend

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="list_collections",
            description=(
                "Return all collection names in the connected MongoDB database. "
                "For small databases (≤50 collections) returns a flat list (mode='flat'). "
                "For larger databases returns collections grouped by name prefix "
                "(mode='grouped'), with a 'standalone' list for ungrouped ones. "
                "Call this first when you don't know which collections exist. "
                "Use search_collections to find collections by name pattern."
            ),
            params=[],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        collections = self._backend.list_collections()
        return ToolResult(success=True, data=_group_collections(collections, self._FLAT_THRESHOLD, self._GROUP_MIN))


# ---------------------------------------------------------------------------
# search_collections
# ---------------------------------------------------------------------------


class SearchCollectionsTool(Tool):
    """Find collections whose names match a glob pattern."""

    def __init__(self, backend: MongoRunner) -> None:
        self._backend = backend

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="search_collections",
            description=(
                "Find collection names that match a glob-style pattern. "
                "Use this when list_collections returns a grouped summary and you need "
                "to locate specific collections by name. "
                "Wildcards: '*' matches any sequence of characters, '?' matches one character. "
                "Examples: 'contest_*' finds all contest collections, '*items*' finds any "
                "collection with 'items' in the name."
            ),
            params=[
                ToolParam(
                    name="pattern",
                    type="string",
                    description="Glob pattern, e.g. 'contest_*', '*items*', 'order?'.",
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern: str = kwargs["pattern"]
        all_collections = self._backend.list_collections()
        matches = sorted(fnmatch.filter(all_collections, pattern))
        return ToolResult(
            success=True,
            data={"pattern": pattern, "matches": matches, "count": len(matches)},
        )


# ---------------------------------------------------------------------------
# describe_collection
# ---------------------------------------------------------------------------


class DescribeCollectionTool(Tool):
    """Return schema, indexes and sample documents for a single collection."""

    def __init__(self, backend: MongoRunner) -> None:
        self._backend = backend

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="describe_collection",
            description=(
                "Return the inferred schema (fields, types, indexes) and sample documents "
                "for a given collection. Use this before writing a query to understand "
                "the data structure."
            ),
            params=[
                ToolParam(
                    name="collection",
                    type="string",
                    description="Name of the collection to describe.",
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        collection: str = kwargs["collection"]
        all_collections = set(self._backend.list_collections())
        schema = self._backend._introspect_collection(collection, all_collections)

        # Serialise FieldInfo list into plain dicts for the LLM.
        def field_to_dict(f: Any) -> dict:
            d: dict = {
                "path": f.path,
                "types": f.types,
                "frequency": f.frequency,
            }
            if f.is_indexed:
                d["indexed"] = True
            if f.is_unique:
                d["unique"] = True
            if f.is_reference:
                d["reference"] = f.reference_collection
            if f.array_element_types:
                d["array_element_types"] = f.array_element_types
            if f.sub_fields:
                d["sub_fields"] = [field_to_dict(sf) for sf in f.sub_fields]
            return d

        payload = {
            "collection": schema.collection_name,
            "document_count": schema.document_count,
            "fields": [field_to_dict(f) for f in schema.fields],
            "indexes": schema.indexes,
            "sample_documents": schema.sample_documents[:3],
        }
        return ToolResult(success=True, data=payload)


# ---------------------------------------------------------------------------
# collection_stats
# ---------------------------------------------------------------------------


class CollectionStatsTool(Tool):
    """Return document counts for all collections, sorted descending."""

    def __init__(self, backend: MongoRunner, max_workers: int = 16) -> None:
        self._backend = backend
        self._max_workers = max_workers

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="collection_stats",
            description=(
                "Return the document count for every collection in the database, "
                "sorted from largest to smallest. "
                "Use this when you need to compare collection sizes or find the "
                "collection with the most (or fewest) documents."
            ),
            params=[
                ToolParam(
                    name="top_n",
                    type="integer",
                    description="Return only the top N collections by document count. Defaults to all.",
                    required=False,
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        top_n: int | None = kwargs.get("top_n")
        names = [n for n in self._backend.list_collections() if not n.startswith("system.")]

        def _count(name: str) -> dict:
            try:
                count = self._backend._database[name].estimated_document_count()
            except Exception:
                count = -1
            return {"collection": name, "document_count": count}

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            stats = sorted(pool.map(_count, names), key=lambda x: x["document_count"], reverse=True)

        if top_n:
            stats = stats[:top_n]

        return ToolResult(success=True, data={"collections": stats, "total": len(names)})


# ---------------------------------------------------------------------------
# run_mql
# ---------------------------------------------------------------------------


class RunMQLTool(Tool):
    """Execute a MQL query and return results as a JSON-serialisable list."""

    def __init__(
        self,
        backend: MongoRunner,
        max_rows: int = 100,
        validate: bool = True,
    ) -> None:
        self._backend = backend
        self._max_rows = max_rows
        self._validator: MQLValidator | None = MQLValidator(backend) if validate else None

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="run_mql",
            description=(
                "Execute a MongoDB query and return the results. "
                "Supports 'find', 'aggregate', 'count', and 'distinct' operations. "
                "Always use 'aggregate' for grouping, sorting pipelines, or $lookup joins. "
                "Results are capped at 100 rows by default."
            ),
            params=[
                ToolParam(
                    name="operation",
                    type="string",
                    description="Query operation type.",
                    enum=["find", "aggregate", "count", "distinct"],
                ),
                ToolParam(
                    name="collection",
                    type="string",
                    description="Name of the target collection.",
                ),
                ToolParam(
                    name="filter",
                    type="object",
                    description="MongoDB filter document (for find/count/distinct).",
                    required=False,
                ),
                ToolParam(
                    name="pipeline",
                    type="array",
                    description=(
                        "Aggregation pipeline stages. "
                        "REQUIRED when operation is 'aggregate' — must be a non-empty list. "
                        "Example: [{\"$group\": {\"_id\": \"$field\", \"count\": {\"$sum\": 1}}}]. "
                        "Omit for find/count/distinct."
                    ),
                    required=False,
                    items={"type": "object"},
                ),
                ToolParam(
                    name="projection",
                    type="object",
                    description="Fields to include/exclude (for find).",
                    required=False,
                ),
                ToolParam(
                    name="sort",
                    type="object",
                    description='Sort specification, e.g. {"created_at": -1}.',
                    required=False,
                ),
                ToolParam(
                    name="limit",
                    type="integer",
                    description=f"Max documents to return (default {100}).",
                    required=False,
                ),
                ToolParam(
                    name="distinct_field",
                    type="string",
                    description="Field name for distinct operation.",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        # Build QueryRequest from kwargs, applying row cap.
        limit = min(int(kwargs.get("limit") or self._max_rows), self._max_rows)

        request = QueryRequest(
            operation=kwargs["operation"],
            collection=kwargs["collection"],
            filter=_coerce_dates(kwargs.get("filter")),
            pipeline=_coerce_dates(_parse_json_arg(kwargs.get("pipeline"))),
            projection=kwargs.get("projection"),
            sort=kwargs.get("sort"),
            limit=limit,
            distinct_field=kwargs.get("distinct_field"),
        )

        warnings: list[str] = []
        if self._validator is not None:
            validation = self._validator.validate(request)
            if not validation.valid:
                return ToolResult(success=False, error=validation.as_tool_error())
            warnings = validation.warnings

        df = self._backend.execute_query(request)

        if df.empty:
            data: dict = {"rows": [], "row_count": 0}
            if warnings:
                data["validation_warnings"] = warnings
            return ToolResult(success=True, data=data)

        rows = json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))
        data = {"rows": rows, "row_count": len(rows)}
        if warnings:
            data["validation_warnings"] = warnings
        return ToolResult(success=True, data=data)


# ---------------------------------------------------------------------------
# explain_query
# ---------------------------------------------------------------------------


class ExplainQueryTool(Tool):
    """Explain a MQL query step-by-step with optional MongoDB execution stats."""

    def __init__(self, backend: MongoRunner, validate: bool = True) -> None:
        self._backend = backend
        self._validator: MQLValidator | None = MQLValidator(backend) if validate else None

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="explain_query",
            description=(
                "Explain what a MongoDB query does and how MongoDB would execute it. "
                "Returns a plain-language breakdown of each pipeline stage or filter, "
                "plus execution statistics (documents examined, index used, time in ms) "
                "when available. "
                "Use this when the user asks to 'explain', 'describe step by step', "
                "'show the query plan', 'walk me through', 'how does this query work?', "
                "'what did you do?', 'why did I get these results?', or 'is this query using an index?'. "
                "Prefer this tool over run_mql when the user wants an explanation rather than raw results."
            ),
            params=[
                ToolParam(
                    name="operation",
                    type="string",
                    description="Query operation type.",
                    enum=["find", "aggregate", "count", "distinct"],
                ),
                ToolParam(
                    name="collection",
                    type="string",
                    description="Name of the target collection.",
                ),
                ToolParam(
                    name="filter",
                    type="object",
                    description="MongoDB filter document (for find/count/distinct).",
                    required=False,
                ),
                ToolParam(
                    name="pipeline",
                    type="array",
                    description=(
                        "Aggregation pipeline stages. "
                        "REQUIRED when operation is 'aggregate' — must be a non-empty list. "
                        "Omit for find/count/distinct."
                    ),
                    required=False,
                    items={"type": "object"},
                ),
                ToolParam(
                    name="projection",
                    type="object",
                    description="Fields to include/exclude (for find).",
                    required=False,
                ),
                ToolParam(
                    name="sort",
                    type="object",
                    description='Sort specification, e.g. {"created_at": -1}.',
                    required=False,
                ),
                ToolParam(
                    name="distinct_field",
                    type="string",
                    description="Field name for distinct operation.",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        request = QueryRequest(
            operation=kwargs["operation"],
            collection=kwargs["collection"],
            filter=_coerce_dates(kwargs.get("filter")),
            pipeline=_coerce_dates(_parse_json_arg(kwargs.get("pipeline"))),
            projection=kwargs.get("projection"),
            sort=kwargs.get("sort"),
            distinct_field=kwargs.get("distinct_field"),
        )

        if self._validator is not None:
            validation = self._validator.validate(request)
            if not validation.valid:
                return ToolResult(success=False, error=validation.as_tool_error())

        # Stage-by-stage logical description (derived from structure, no DB call).
        stages = _describe_request(request)

        # Execution stats from MongoDB .explain() — best-effort, not all backends support it.
        exec_stats: dict = {}
        try:
            exec_stats = _run_explain(self._backend, request)
        except Exception:
            pass

        data: dict = {
            "collection": request.collection,
            "operation": request.operation,
            "stages": stages,
        }
        if exec_stats:
            data["execution_stats"] = exec_stats

        return ToolResult(success=True, data=data)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_json_arg(value: Any) -> Any:
    """Parse a JSON string into a Python object, or return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_dates(obj: Any) -> Any:
    """Recursively convert ISO date strings and {$date: ...} objects to datetime.

    The LLM often generates date filters as plain strings ("2024-01-01T00:00:00")
    or Extended JSON ({$date: "..."}). MongoDB requires actual datetime objects
    when the field is stored as BSON date type. This function normalises both forms.
    """
    if obj is None:
        return None

    if isinstance(obj, list):
        return [_coerce_dates(item) for item in obj]

    if isinstance(obj, dict):
        # Extended JSON form: {"$date": "2024-01-01T00:00:00.000Z"}
        if list(obj.keys()) == ["$date"]:
            return _parse_iso(obj["$date"])
        return {k: _coerce_dates(v) for k, v in obj.items()}

    if isinstance(obj, str) and _ISO_DATE_RE.match(obj):
        parsed = _parse_iso(obj)
        # Only substitute if we successfully parsed a datetime.
        return parsed if parsed is not None else obj

    return obj


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO 8601 string to datetime, returning None on failure."""
    if not isinstance(value, str):
        return None
    # Normalise: remove trailing Z, replace space separator with T.
    s = value.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try progressively shorter formats.
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            # Strip timezone info — pymongo handles naive datetimes as UTC.
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Explain helpers
# ---------------------------------------------------------------------------


def _describe_request(request: QueryRequest) -> list[dict]:
    """Return a plain-language description of each logical step in *request*."""
    stages: list[dict] = []

    if request.operation in ("find", "count"):
        if request.filter:
            stages.append({
                "step": "filter",
                "description": f"Filter where: {json.dumps(request.filter, default=str)}",
            })
        else:
            stages.append({"step": "filter", "description": "No filter — all documents"})

        if request.sort:
            parts = [f"{k} {'desc' if v == -1 else 'asc'}" for k, v in request.sort.items()]
            stages.append({"step": "sort", "description": f"Sort: {', '.join(parts)}"})

        if request.projection:
            included = [k for k, v in request.projection.items() if v and k != "_id"]
            excluded = [k for k, v in request.projection.items() if not v]
            if included:
                stages.append({"step": "projection", "description": f"Return fields: {', '.join(included)}"})
            elif excluded:
                stages.append({"step": "projection", "description": f"Exclude fields: {', '.join(excluded)}"})

        if request.operation == "count":
            stages.append({"step": "count", "description": "Count matching documents"})

    elif request.operation == "distinct":
        stages.append({
            "step": "distinct",
            "description": f"Unique values of '{request.distinct_field}'"
            + (f" where: {json.dumps(request.filter, default=str)}" if request.filter else ""),
        })

    elif request.operation == "aggregate" and request.pipeline:
        for i, stage in enumerate(request.pipeline):
            if not isinstance(stage, dict) or len(stage) != 1:
                stages.append({"step": f"stage_{i}", "description": str(stage)})
                continue
            op = next(iter(stage))
            body = stage[op]
            stages.append({"step": op, "description": _describe_stage(op, body)})

    return stages


def _describe_stage(op: str, body: Any) -> str:
    """Return a one-line plain-language description of a single pipeline stage."""
    if op == "$match":
        return f"Filter: {json.dumps(body, default=str)}"

    if op == "$group":
        id_expr = json.dumps(body.get("_id"), default=str)
        computed = [k for k in body if k != "_id"]
        desc = f"Group by {id_expr}"
        if computed:
            desc += f", compute: {', '.join(computed)}"
        return desc

    if op == "$sort":
        if isinstance(body, dict):
            parts = [f"{k} {'desc' if v == -1 else 'asc'}" for k, v in body.items()]
            return f"Sort: {', '.join(parts)}"
        return f"Sort: {body}"

    if op == "$limit":
        return f"Keep first {body} documents"

    if op == "$skip":
        return f"Skip {body} documents"

    if op == "$project":
        if isinstance(body, dict):
            included = [k for k, v in body.items() if v and k != "_id"]
            excluded = [k for k, v in body.items() if not v]
            if included:
                return f"Return fields: {', '.join(included)}"
            if excluded:
                return f"Exclude fields: {', '.join(excluded)}"
        return f"Project: {json.dumps(body, default=str)[:100]}"

    if op in ("$addFields", "$set"):
        fields = list(body.keys()) if isinstance(body, dict) else []
        return f"Add/compute fields: {', '.join(fields)}"

    if op in ("$unset", "$project"):
        fields = body if isinstance(body, list) else [body]
        return f"Remove fields: {', '.join(str(f) for f in fields)}"

    if op == "$unwind":
        field = body if isinstance(body, str) else body.get("path", str(body))
        preserve = body.get("preserveNullAndEmptyArrays", False) if isinstance(body, dict) else False
        desc = f"Expand array: {field}"
        if preserve:
            desc += " (keep nulls)"
        return desc

    if op == "$lookup":
        if isinstance(body, dict):
            return (
                f"Join '{body.get('from')}' on "
                f"{body.get('localField')} → {body.get('foreignField')}"
                + (f", as '{body.get('as')}'" if body.get("as") else "")
            )

    if op == "$count":
        return f"Count documents → '{body}'"

    if op == "$sample":
        size = body.get("size") if isinstance(body, dict) else body
        return f"Random sample of {size} documents"

    if op == "$bucket":
        if isinstance(body, dict):
            return f"Bucket '{body.get('groupBy')}' by boundaries {body.get('boundaries')}"

    if op == "$facet":
        if isinstance(body, dict):
            return f"Multi-facet analysis: {', '.join(body.keys())}"

    if op in ("$out", "$merge"):
        into = body.get("into", body) if isinstance(body, dict) else body
        return f"Write results to '{into}'"

    if op in ("$replaceRoot", "$replaceWith"):
        return f"Replace root document with: {json.dumps(body, default=str)[:80]}"

    if op == "$sortByCount":
        return f"Sort by count of '{body}'"

    # Fallback for any other stage
    return f"{op}: {json.dumps(body, default=str)[:100]}"


def _run_explain(backend: MongoRunner, request: QueryRequest) -> dict:
    """Run MongoDB .explain() and return a normalised stats dict.

    Returns an empty dict when the backend does not support explain
    (e.g. mongomock) or when the operation is not supported.
    """
    db = backend._database
    col = db[request.collection]

    try:
        if request.operation in ("find", "count"):
            cursor = col.find(request.filter or {})
            if request.sort:
                cursor = cursor.sort(list(request.sort.items()))
            raw = cursor.explain("executionStats")

        elif request.operation == "aggregate":
            raw = db.command(
                "aggregate",
                request.collection,
                pipeline=request.pipeline,
                explain=True,
                cursor={},
            )

        elif request.operation == "distinct":
            raw = db.command(
                "distinct",
                request.collection,
                key=request.distinct_field or "",
                query=request.filter or {},
                explain=True,
            )

        else:
            return {}

    except Exception:
        return {}

    return _parse_explain(raw)


def _parse_explain(raw: dict) -> dict:
    """Extract useful fields from a raw MongoDB explain document."""
    stats: dict = {}

    exec_stats = raw.get("executionStats", {})
    if exec_stats:
        for key, alias in (
            ("nReturned", "docs_returned"),
            ("totalDocsExamined", "docs_examined"),
            ("totalKeysExamined", "keys_examined"),
            ("executionTimeMillis", "execution_time_ms"),
        ):
            val = exec_stats.get(key)
            if val is not None:
                stats[alias] = val

        # Scan type from the winning plan (IXSCAN / COLLSCAN / etc.)
        winning_plan = raw.get("queryPlanner", {}).get("winningPlan", {})
        scan_stage = winning_plan.get("inputStage", winning_plan)
        scan_type = scan_stage.get("stage", "")
        if scan_type:
            stats["scan_type"] = scan_type
            if scan_type == "IXSCAN":
                key_pattern = scan_stage.get("keyPattern")
                if key_pattern:
                    stats["index_used"] = str(key_pattern)

    # Aggregate explain has a "stages" list instead of executionStats.
    agg_stages = raw.get("stages")
    if agg_stages and not stats:
        stats["pipeline_stages_executed"] = len(agg_stages)

    return stats


# ---------------------------------------------------------------------------
# Collection grouping helper
# ---------------------------------------------------------------------------


def _group_collections(
    collections: list[str],
    flat_threshold: int = 50,
    group_min: int = 3,
) -> dict:
    """Return a grouped or flat summary of collection names for the LLM.

    - If len(collections) <= flat_threshold: returns mode='flat'.
    - Otherwise: groups by UUID prefix, then by shared word prefix,
      and lists remaining as standalone. Returns mode='grouped'.
    """
    if len(collections) <= flat_threshold:
        return {"mode": "flat", "collections": sorted(collections), "count": len(collections)}

    groups: dict[str, list[str]] = {}
    no_uuid: list[str] = []

    # --- Pass 1: group collections that contain a UUID in their name ---
    for name in collections:
        match = _UUID_RE.search(name)
        if match:
            prefix_raw = name[: match.start()].rstrip("_- ")
            prefix = prefix_raw if prefix_raw else "<uuid>"
            groups.setdefault(prefix, []).append(name)
        else:
            no_uuid.append(name)

    # --- Pass 2: group remaining collections by shared word prefix ---
    prefix_hits: dict[str, list[str]] = {}
    for name in no_uuid:
        parts = name.split("_")
        for length in range(1, len(parts)):
            candidate = "_".join(parts[:length])
            prefix_hits.setdefault(candidate, []).append(name)

    assigned: set[str] = set()
    for candidate in sorted(prefix_hits, key=lambda p: -len(p)):
        members = [m for m in prefix_hits[candidate] if m not in assigned]
        if len(members) >= group_min:
            groups.setdefault(candidate, []).extend(members)
            assigned.update(members)

    standalone = sorted(n for n in no_uuid if n not in assigned)

    group_list = [
        {"prefix": f"{prefix}_*", "count": len(members), "example": members[0]}
        for prefix, members in sorted(groups.items())
    ]

    return {
        "mode": "grouped",
        "total_collections": len(collections),
        "groups": group_list,
        "standalone": standalone,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_mongo_tools(
    backend: MongoRunner,
    max_rows: int = 100,
    validate: bool = True,
) -> list[Tool]:
    """Return the standard set of MongoDB tools ready to register."""
    return [
        ListCollectionsTool(backend),
        SearchCollectionsTool(backend),
        DescribeCollectionTool(backend),
        CollectionStatsTool(backend),
        RunMQLTool(backend, max_rows=max_rows, validate=validate),
    ]
