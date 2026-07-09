"""Tests for mango.agent.prompt_builder."""

from __future__ import annotations

import pytest

from mango.agent.prompt_builder import build_system_prompt, format_memory_examples, value_hints_section
from mango.core.types import FieldInfo, SchemaInfo
from mango.memory.models import MemoryEntry
from mango.integrations.chromadb import make_entry_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(collection_name: str, doc_count: int = 10) -> SchemaInfo:
    return SchemaInfo(
        collection_name=collection_name,
        document_count=doc_count,
        fields=[
            FieldInfo(name="_id", path="_id", types=["ObjectId"], frequency=1.0),
            FieldInfo(name="name", path="name", types=["string"], frequency=1.0),
        ],
        indexes=[],
        sample_documents=[],
    )


def _make_memory_entry(question: str, similarity: float = 0.85) -> MemoryEntry:
    return MemoryEntry(
        id=make_entry_id(),
        question=question,
        tool_name="run_mql",
        tool_args={"operation": "count", "collection": "users"},
        result_summary="42 users found.",
        similarity=similarity,
    )


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_returns_string(self):
        prompt = build_system_prompt(db_name="mydb")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_db_name(self):
        prompt = build_system_prompt(db_name="mydb")
        assert "mydb" in prompt

    def test_includes_list_collections_instruction(self):
        prompt = build_system_prompt(db_name="mydb")
        assert "list_collections" in prompt

    def test_includes_role_description(self):
        prompt = build_system_prompt(db_name="mydb")
        assert "Mango" in prompt

    def test_includes_read_only_rule(self):
        prompt = build_system_prompt(db_name="mydb")
        assert "NEVER" in prompt
        assert "write" in prompt.lower()

    def test_read_only_rule_forbids_suggesting_write_commands_as_text(self):
        # Regression guard: the tool-call layer already blocks writes
        # (MQLValidator/security.py), but nothing scrubs the free-text final
        # answer. Without this clause the agent can satisfy "never perform a
        # write" while still handing the user a ready-to-run updateMany/
        # deleteMany/$out snippet in prose, defeating the read-only guarantee.
        prompt = build_system_prompt(db_name="mydb").lower()
        assert "suggest" in prompt or "hand the user" in prompt
        assert "updatemany" in prompt or "deletemany" in prompt

    def test_without_schema_no_schema_section(self):
        prompt = build_system_prompt(db_name="mydb", schema=None)
        assert "## Schema" not in prompt

    def test_with_schema_small_includes_full_details(self):
        schema = {"users": _make_schema("users")}
        prompt = build_system_prompt(db_name="mydb", schema=schema)
        assert "## Schema" in prompt
        assert "name" in prompt

    def test_with_schema_large_shows_summary(self):
        # More than 10 collections → summary mode.
        collections = [f"col{i}" for i in range(15)]
        schema = {c: _make_schema(c) for c in collections}
        prompt = build_system_prompt(db_name="mydb", schema=schema)
        assert "## Schema" in prompt
        assert "describe_collection" in prompt

    def test_includes_output_format_section(self):
        prompt = build_system_prompt(db_name="mydb")
        assert "Output format" in prompt


# ---------------------------------------------------------------------------
# format_memory_examples
# ---------------------------------------------------------------------------


class TestFormatMemoryExamples:
    def test_empty_list_returns_empty_string(self):
        assert format_memory_examples([]) == ""

    def test_single_example_included(self):
        entry = _make_memory_entry("How many users are there?")
        result = format_memory_examples([entry])
        assert "How many users are there?" in result
        assert "run_mql" in result
        assert "42 users found." in result

    def test_similarity_shown_as_percentage(self):
        entry = _make_memory_entry("Count users", similarity=0.75)
        result = format_memory_examples([entry])
        assert "75%" in result

    def test_multiple_examples_numbered(self):
        entries = [
            _make_memory_entry("Question A"),
            _make_memory_entry("Question B"),
        ]
        result = format_memory_examples(entries)
        assert "Example 1" in result
        assert "Example 2" in result

    def test_tool_args_serialised_as_json(self):
        entry = _make_memory_entry("How many?")
        result = format_memory_examples([entry])
        assert "count" in result  # from tool_args
        assert "users" in result


# ---------------------------------------------------------------------------
# value_hints_section
# ---------------------------------------------------------------------------


class TestValueHintsSection:
    def test_empty_list_returns_empty_string(self):
        assert value_hints_section([]) == ""

    def test_hints_included(self):
        hints = ["- \"active\" -> in `orders.status` it is stored as 'ACTIVE'"]
        result = value_hints_section(hints)
        assert "orders.status" in result
        assert "ACTIVE" in result

    def test_header_present(self):
        result = value_hints_section(["- some hint"])
        assert "## Value hints" in result

    def test_mentions_inspect_field_is_unnecessary(self):
        result = value_hints_section(["- some hint"])
        assert "inspect_field" in result
