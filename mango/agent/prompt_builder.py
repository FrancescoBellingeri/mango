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
        "NEVER perform write operations (insert, update, delete, drop). Read-only only.",
        "ALWAYS call describe_collection before writing a query for a collection you "
        "haven't inspected yet in this conversation.",
        "If a query returns no results, explain why (wrong filter, empty collection, etc.).",
        "If the question is ambiguous, ask one clarifying question before querying.",
        "Limit results to a sensible number (≤100 rows). Never fetch unbounded results.",
        "When filtering on date fields, check the actual format in the sample documents first. "
        "Dates may be stored as strings without timezone (e.g. '2024-11-22T08:29:28.225') — "
        "match the exact format, do not add 'Z' or timezone suffixes unless present in the data.",
        "Prefer aggregate pipelines over multiple find queries when joining or grouping.",
        "If a query fails, analyse the error and retry once with a corrected query.",
        "At the START of every question, call search_saved_correct_tool_uses to find similar "
        "past interactions. Use the results to guide your tool selection and arguments.",
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
