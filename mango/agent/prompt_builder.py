"""System prompt builder.

Builds the system prompt injected into every LLM conversation.
The prompt includes:
  - Role and behavioural rules
  - Connected database name and collection list
  - Schema summary for relevant collections (or all if few enough)
"""

from __future__ import annotations

import json
from mango.core.types import FieldInfo, SchemaInfo
from mango.memory import MemoryEntry


# Max number of collections to include full schema for in the system prompt.
# Beyond this, only names are listed to save tokens.
_FULL_SCHEMA_THRESHOLD = 10

# Max fields per collection to render in detail (avoid token explosion).
_MAX_FIELDS_PER_COLLECTION = 40


def build_system_prompt(
    db_name: str,
    schema: dict[str, SchemaInfo] | None = None,
) -> str:
    """Build the system prompt for the Mango agent.

    Args:
        db_name: Name of the connected MongoDB database.
        schema: Optional pre-introspected schema map. When provided, field
                details are included in the prompt.

    Returns:
        System prompt string ready to pass to LLMService.chat().
    """
    sections: list[str] = []

    sections.append(_role_section())
    sections.append(_rules_section())
    sections.append(_database_section(db_name))

    if schema:
        sections.append(_schema_section(schema))

    sections.append(_output_section())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _role_section() -> str:
    return (
        "You are Mango, an AI assistant that translates natural language questions "
        "into MongoDB queries and returns clear, accurate answers.\n"
        "You have access to tools to inspect the database schema and run queries. "
        "Always prefer using tools over guessing."
    )


def _rules_section() -> str:
    rules = [
        "NEVER perform write operations (insert, update, delete, drop). Read-only only. "
        "This also means: never write out a write command/query/shell snippet (e.g. updateMany, "
        "deleteMany, $out, $merge) anywhere in your answer, in ANY framing — not as a suggestion, "
        "not 'for reference', not as 'the command would be', not in a code block 'just to show "
        "what it would look like'. Describing the write query is the same violation as running "
        "it: the user can copy-paste either one. If the request requires modifying data, say "
        "plainly that this is a read-only assistant and the change must be made through a proper "
        "write-access channel — full stop, with no query, pseudocode, or code block for it.",
        "ALWAYS call describe_collection before writing a query for a collection you "
        "haven't inspected yet in this conversation.",
        "If a query returns no results, explain why (wrong filter, empty collection, etc.).",
        "If the question is ambiguous, ask one clarifying question before querying.",
        "Limit results to a sensible number (≤100 rows). Never fetch unbounded results.",
        "If a '## Value hints' section is present below, it already tells you the exact "
        "stored encoding for values mentioned in the question — use it directly in your "
        "filter and do NOT call inspect_field for that field. "
        "Otherwise, before filtering or grouping on a categorical field by a specific value, call "
        "inspect_field to confirm how its values are actually encoded — exact spelling, "
        "casing and data type — instead of assuming. If the same concept appears under "
        "more than one form, match all of them (e.g. $in, or a case-insensitive $regex). "
        "inspect_field is only a diagnostic to inform the filter: it is capped and may be "
        "sampled, so NEVER answer from its output — always compute the final answer by "
        "executing the query with run_mql.",
        "When filtering on date fields, check the actual format in the sample documents first. "
        "Dates may be stored as strings without timezone (e.g. '2024-11-22T08:29:28.225') — "
        "match the exact format, do not add 'Z' or timezone suffixes unless present in the data.",
        "Prefer aggregate pipelines over multiple find queries when joining or grouping.",
        "If a query fails, analyse the error and retry once with a corrected query.",
        "After run_mql returns a result, verify it actually answers the question: "
        "check that (1) the row count is plausible, (2) the output fields match what was asked, "
        "(3) numeric values are in a reasonable range. "
        "If the result looks wrong or incomplete, re-query with a corrected approach before answering.",
        "Whenever you discover something useful about the database structure or business meaning "
        "(e.g. which collection holds customer data, what a field's values mean), "
        "call save_text_memory to persist that insight.",
    ]
    lines = "\n".join(f"- {r}" for r in rules)
    return f"## Rules\n{lines}"


def _database_section(db_name: str) -> str:
    return (
        f"## Connected database\n"
        f"Database name: `{db_name}`\n"
        f"Use list_collections to explore available collections."
    )


def _schema_section(schema: dict[str, SchemaInfo]) -> str:
    collections = list(schema.keys())
    lines: list[str] = ["## Schema"]

    if len(collections) > _FULL_SCHEMA_THRESHOLD:
        lines.append(
            f"There are {len(collections)} collections. "
            "Full schema is available via describe_collection. "
            "Below is a summary of each collection."
        )

    for name in collections:
        info = schema.get(name)
        if info is None:
            lines.append(f"\n### {name}\n_(schema not available)_")
            continue

        lines.append(f"\n### {name}")
        lines.append(f"Documents: ~{info.document_count:,}")

        if len(collections) <= _FULL_SCHEMA_THRESHOLD:
            lines.append(_render_fields(info.fields))
        else:
            top_level = [f for f in info.fields if "." not in f.path][:15]
            brief = ", ".join(
                f"`{f.name}` ({'/'.join(f.types)})" for f in top_level
            )
            lines.append(f"Fields: {brief}")

    return "\n".join(lines)


def schema_section_for_query(
    schema: dict[str, SchemaInfo],
    collection_names: list[str],
    total_collections: int,
) -> str:
    """Build schema section for a specific subset of collections (per-query injection).

    Always lists ALL collection names so the agent knows what is available,
    then shows full field details only for the *collection_names* subset.
    """
    all_names = sorted(schema.keys())
    all_names_str = ", ".join(f"`{n}`" for n in all_names)

    # Show all names only when not all schemas are already fully shown, so we
    # don't bloat the prompt for small databases that show everything anyway.
    show_all_header = len(collection_names) < total_collections
    lines: list[str] = []
    if show_all_header:
        lines.append(f"## Available collections ({total_collections} total): {all_names_str}")
        lines.append(
            f"\n## Relevant schema — full details for {len(collection_names)} collection(s) "
            "(use describe_collection for the others)"
        )
    else:
        lines.append(f"## Schema ({total_collections} collection(s))")
    for name in collection_names:
        info = schema.get(name)
        if info is None:
            lines.append(f"\n### {name}\n_(schema not available)_")
            continue
        lines.append(f"\n### {name}")
        lines.append(f"Documents: ~{info.document_count:,}")
        lines.append(_render_fields(info.fields))
    return "\n".join(lines)


def _output_section() -> str:
    return (
        "## Output format\n"
        "- Answer in the same language the user uses.\n"
        "- After running a query, summarise the results in plain language.\n"
        "- Show raw data only when the user explicitly asks for it.\n"
        "- Use markdown tables for tabular results when appropriate."
    )


# ---------------------------------------------------------------------------
# Field rendering
# ---------------------------------------------------------------------------


def _render_fields(fields: list[FieldInfo], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = "  " * indent

    for f in fields[:_MAX_FIELDS_PER_COLLECTION]:
        type_str = "/".join(f.types)
        freq = f" ({f.frequency:.0%})" if f.frequency < 1.0 else ""
        flags: list[str] = []
        if f.is_indexed:
            flags.append("indexed")
        if f.is_unique:
            flags.append("unique")
        if f.is_reference:
            flags.append(f"→ {f.reference_collection}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        lines.append(f"{prefix}- `{f.path}`: {type_str}{freq}{flag_str}")

        if f.sub_fields:
            lines.append(_render_fields(f.sub_fields, indent + 1))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Value-grounding hints
# ---------------------------------------------------------------------------


def value_hints_section(hints: list[str]) -> str:
    """Render proactive value-grounding hints for the current question.

    Returns an empty string when there are no hints — callers should skip the
    section entirely rather than emit an empty header.
    """
    if not hints:
        return ""
    lines = [
        "## Value hints for this question",
        "Exact stored encoding for values mentioned above — use these directly "
        "in your filter, no need to call inspect_field for them:",
    ]
    lines.extend(hints)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory examples
# ---------------------------------------------------------------------------


def format_memory_examples(examples: list[MemoryEntry]) -> str:
    """Format retrieved memory entries as few-shot examples for the prompt."""
    if not examples:
        return ""

    lines = ["## Similar past interactions (use as reference)"]
    for i, ex in enumerate(examples, 1):
        args_str = json.dumps(ex.tool_args, ensure_ascii=False)
        lines.append(
            f"\n### Example {i} (similarity: {ex.similarity:.0%})\n"
            f"**Question:** {ex.question}\n"
            f"**Tool used:** `{ex.tool_name}`\n"
            f"**Arguments:** `{args_str}`\n"
            f"**Result summary:** {ex.result_summary}"
        )

    return "\n".join(lines)
