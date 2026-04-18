"""Pre-execution MQL validator.

Validates a QueryRequest for structural correctness before it hits the
database, catching LLM hallucinations early.

Validation is split into two severity levels:
  - Error   → blocks execution, returned as ToolResult(success=False)
  - Warning → informational, logged alongside a successful result

Checks performed:
  1. Operation in allowlist (find/aggregate/count/distinct)
  2. Collection exists in the database
  3. Required arguments per operation (pipeline for aggregate, etc.)
  4. Pipeline stage names are valid MQL aggregation stages
  5. $ operators in filter / pipeline are recognised MQL operators
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from mango.core.types import QueryRequest
from mango.nosql_runner import NoSQLRunner


# ---------------------------------------------------------------------------
# Known MQL identifiers
# ---------------------------------------------------------------------------

_ALLOWED_OPERATIONS = frozenset({"find", "aggregate", "count", "distinct"})

# Filter/query operators (used in filter, $match, $expr, etc.)
_FILTER_OPERATORS = frozenset({
    # Comparison
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
    # Logical
    "$and", "$or", "$nor", "$not",
    # Element
    "$exists", "$type",
    # Evaluation
    "$regex", "$options", "$expr", "$jsonSchema", "$mod", "$text", "$where",
    # Array
    "$all", "$elemMatch", "$size",
    # Bitwise
    "$bitsAllClear", "$bitsAllSet", "$bitsAnyClear", "$bitsAnySet",
    # Geospatial
    "$geoIntersects", "$geoWithin", "$near", "$nearSphere",
    "$geometry", "$maxDistance", "$minDistance", "$box", "$center",
    "$centerSphere", "$polygon",
    # Meta
    "$comment", "$rand",
})

# Aggregation pipeline stage operators
_PIPELINE_STAGES = frozenset({
    "$addFields", "$bucket", "$bucketAuto", "$changeStream", "$collStats",
    "$count", "$densify", "$documents", "$facet", "$fill", "$geoNear",
    "$graphLookup", "$group", "$indexStats", "$limit", "$listLocalSessions",
    "$listSessions", "$lookup", "$match", "$merge", "$out", "$planCacheStats",
    "$project", "$redact", "$replaceRoot", "$replaceWith", "$sample",
    "$search", "$searchMeta", "$set", "$setWindowFields", "$skip", "$sort",
    "$sortByCount", "$unionWith", "$unset", "$unwind", "$vectorSearch",
})

# Accumulator operators (used in $group, $bucket, $setWindowFields, etc.)
_ACCUMULATOR_OPERATORS = frozenset({
    "$addToSet", "$avg", "$bottom", "$bottomN", "$count", "$first", "$firstN",
    "$last", "$lastN", "$max", "$maxN", "$median", "$mergeObjects", "$min",
    "$minN", "$percentile", "$push", "$stdDevPop", "$stdDevSamp", "$sum",
    "$top", "$topN", "$accumulator", "$function",
})

# Arithmetic / string / date / array expression operators
_EXPRESSION_OPERATORS = frozenset({
    "$abs", "$add", "$allElementsTrue", "$anyElementTrue", "$arrayElemAt",
    "$arrayToObject", "$ceil", "$cmp", "$concat", "$concatArrays", "$cond",
    "$convert", "$dateAdd", "$dateDiff", "$dateFromParts", "$dateFromString",
    "$dateToParts", "$dateToString", "$dateTrunc", "$dayOfMonth", "$dayOfWeek",
    "$dayOfYear", "$divide", "$exp", "$filter", "$floor", "$getField",
    "$hour", "$ifNull", "$indexOfArray", "$indexOfBytes", "$indexOfCP",
    "$isArray", "$isNumber", "$isoDayOfWeek", "$isoWeek", "$isoWeekYear",
    "$let", "$literal", "$log", "$log10", "$ltrim", "$map", "$meta",
    "$millisecond", "$minute", "$mod", "$month", "$multiply", "$objectToArray",
    "$pow", "$range", "$reduce", "$regexFind", "$regexFindAll", "$regexMatch",
    "$replaceAll", "$replaceOne", "$reverseArray", "$rtrim", "$second",
    "$setDifference", "$setEquals", "$setIntersection", "$setIsSubset",
    "$setUnion", "$slice", "$split", "$sqrt", "$strcasecmp", "$strLenBytes",
    "$strLenCP", "$substr", "$substrBytes", "$substrCP", "$subtract",
    "$switch", "$toBool", "$toDate", "$toDecimal", "$toDouble", "$toInt",
    "$toLong", "$toLower", "$toObjectId", "$toString", "$toUpper", "$trim",
    "$trunc", "$type", "$unsetField", "$week", "$year", "$zip",
})

# Union of all known $ identifiers — used for unknown-operator detection.
_ALL_KNOWN_OPERATORS: frozenset[str] = (
    _FILTER_OPERATORS
    | _PIPELINE_STAGES
    | _ACCUMULATOR_OPERATORS
    | _EXPRESSION_OPERATORS
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a validation run.

    Attributes:
        valid: False when at least one error is present. Warnings alone
            do not set valid=False.
        errors: Blocking issues — execution should not proceed.
        warnings: Non-blocking observations — execution may proceed.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_tool_error(self) -> str:
        """Format as a human-readable error string for ToolResult."""
        lines = ["Query validation failed:"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class MQLValidator:
    """Validates a QueryRequest before it is sent to the database.

    Args:
        backend: Connected NoSQL backend used for collection listing.

    Checks:
        - Operation is in the read-only allowlist.
        - Target collection exists (suggests similar names on failure).
        - Required arguments are present (pipeline for aggregate, etc.).
        - All aggregation pipeline stage keys are valid MQL stage names.
        - All $ operators in filter/pipeline are recognised MQL operators.
    """

    def __init__(self, backend: NoSQLRunner) -> None:
        self._backend = backend

    def validate(self, request: QueryRequest) -> ValidationResult:
        """Run all validation checks against *request*.

        Args:
            request: The query request to validate.

        Returns:
            ValidationResult with errors (blocking) and warnings
            (informational). valid=False when any error is present.
        """
        errors: list[str] = []
        warnings: list[str] = []

        self._check_operation(request, errors)
        self._check_collection(request, errors)
        self._check_required_args(request, errors)
        self._check_pipeline_stages(request, errors)
        self._check_operators(request, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_operation(self, request: QueryRequest, errors: list[str]) -> None:
        if request.operation not in _ALLOWED_OPERATIONS:
            errors.append(
                f"Operation '{request.operation}' is not allowed. "
                f"Allowed operations: {sorted(_ALLOWED_OPERATIONS)}."
            )

    def _check_collection(self, request: QueryRequest, errors: list[str]) -> None:
        try:
            known = self._backend.list_collections()
        except Exception:
            return  # Can't list — skip gracefully rather than blocking.
        if request.collection not in known:
            msg = f"Collection '{request.collection}' does not exist."
            suggestions = difflib.get_close_matches(
                request.collection, known, n=3, cutoff=0.6
            )
            if suggestions:
                msg += f" Did you mean: {suggestions}?"
            errors.append(msg)

    def _check_required_args(self, request: QueryRequest, errors: list[str]) -> None:
        if request.operation == "aggregate" and not request.pipeline:
            errors.append(
                "Operation 'aggregate' requires a non-empty 'pipeline' argument."
            )
        if request.operation == "distinct" and not request.distinct_field:
            errors.append(
                "Operation 'distinct' requires a 'distinct_field' argument."
            )

    def _check_pipeline_stages(
        self, request: QueryRequest, errors: list[str]
    ) -> None:
        if not request.pipeline:
            return
        for i, stage in enumerate(request.pipeline):
            if not isinstance(stage, dict):
                errors.append(
                    f"Pipeline stage at index {i} must be a dict, got {type(stage).__name__}."
                )
                continue
            if len(stage) != 1:
                keys = list(stage.keys())
                errors.append(
                    f"Pipeline stage at index {i} has {len(keys)} keys: {keys}. "
                    f"Each stage must be a single-key dict mapping the operator to its argument, "
                    f"e.g. {{\"$match\": {{...}}}} or {{\"$group\": {{\"_id\": \"$field\"}}}}."
                )
                continue
            stage_op = next(iter(stage))
            if stage_op not in _PIPELINE_STAGES:
                suggestions = difflib.get_close_matches(
                    stage_op, _PIPELINE_STAGES, n=2, cutoff=0.7
                )
                hint = f" Did you mean {suggestions}?" if suggestions else ""
                errors.append(
                    f"Unknown aggregation stage '{stage_op}' at pipeline index {i}.{hint}"
                )

    def _check_operators(
        self, request: QueryRequest, warnings: list[str]
    ) -> None:
        unknown: list[str] = []

        if request.filter:
            _collect_unknown_operators(request.filter, unknown)

        if request.pipeline:
            for stage in request.pipeline:
                if isinstance(stage, dict):
                    # Skip the stage key itself (validated above); check the body.
                    for body in stage.values():
                        _collect_unknown_operators(body, unknown)

        seen: set[str] = set()
        for op in unknown:
            if op not in seen:
                seen.add(op)
                warnings.append(
                    f"Unrecognised operator '{op}' — verify this is a valid MQL operator."
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_unknown_operators(obj: Any, unknown: list[str]) -> None:
    """Recursively collect $ keys not present in _ALL_KNOWN_OPERATORS."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("$") and k not in _ALL_KNOWN_OPERATORS:
                unknown.append(k)
            _collect_unknown_operators(v, unknown)
    elif isinstance(obj, list):
        for item in obj:
            _collect_unknown_operators(item, unknown)
