"""Tests for mango.agent.value_grounding — proactive value-grounding hints."""

from __future__ import annotations

from mango.agent.value_grounding import build_value_index, find_value_hints
from mango.core.types import FieldInfo, SchemaInfo


def _schema(collection: str, fields: list[FieldInfo]) -> dict[str, SchemaInfo]:
    return {
        collection: SchemaInfo(
            collection_name=collection,
            document_count=10,
            fields=fields,
            indexes=[],
            sample_documents=[],
        )
    }


# ---------------------------------------------------------------------------
# build_value_index
# ---------------------------------------------------------------------------


class TestBuildValueIndex:
    def test_indexes_top_level_field(self):
        schema = _schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["active", "inactive"])],
        )
        index = build_value_index(schema)
        assert index["active"] == [("orders", "status", "active")]

    def test_ignores_field_without_sample_values(self):
        schema = _schema(
            "orders",
            [FieldInfo(name="notes", path="notes", types=["string"], frequency=1.0)],
        )
        index = build_value_index(schema)
        assert index == {}

    def test_indexes_nested_subfield(self):
        country_field = FieldInfo(
            name="country", path="address.country", types=["string"], frequency=1.0,
            sample_values=["Italy"],
        )
        schema = _schema(
            "users",
            [FieldInfo(name="address", path="address", types=["subdocument"], frequency=1.0,
                       sub_fields=[country_field])],
        )
        index = build_value_index(schema)
        assert index["italy"] == [("users", "address.country", "Italy")]

    def test_multiple_variants_share_normalized_key(self):
        schema = _schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["active", "ACTIVE"])],
        )
        index = build_value_index(schema)
        assert set(index["active"]) == {
            ("orders", "status", "active"),
            ("orders", "status", "ACTIVE"),
        }


# ---------------------------------------------------------------------------
# find_value_hints
# ---------------------------------------------------------------------------


class TestFindValueHints:
    def test_empty_index_returns_no_hints(self):
        assert find_value_hints("how many active orders?", {}) == []

    def test_exact_match_no_drift_produces_no_hint(self):
        index = build_value_index(_schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["active"])],
        ))
        hints = find_value_hints("how many active orders?", index)
        assert hints == []

    def test_casing_drift_produces_hint(self):
        index = build_value_index(_schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["ACTIVE"])],
        ))
        hints = find_value_hints("how many active orders?", index)
        assert len(hints) == 1
        assert "orders.status" in hints[0]
        assert "ACTIVE" in hints[0]

    def test_multiple_forms_lists_all_variants(self):
        index = build_value_index(_schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["active", "ACTIVE"])],
        ))
        hints = find_value_hints("how many active orders?", index)
        assert len(hints) == 1
        assert "active" in hints[0] and "ACTIVE" in hints[0]
        assert "$in" in hints[0]

    def test_fuzzy_typo_match(self):
        index = build_value_index(_schema(
            "restaurants",
            [FieldInfo(name="cuisine", path="cuisine", types=["string"], frequency=1.0,
                       sample_values=["Italian"])],
        ))
        hints = find_value_hints("show me italain restaurants", index)
        assert len(hints) == 1
        assert "Italian" in hints[0]

    def test_no_match_returns_empty(self):
        index = build_value_index(_schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["active", "inactive"])],
        ))
        hints = find_value_hints("what is the total revenue?", index)
        assert hints == []

    def test_multi_word_value_matched_via_bigram(self):
        index = build_value_index(_schema(
            "restaurants",
            [FieldInfo(name="city", path="city", types=["string"], frequency=1.0,
                       sample_values=["New York"])],
        ))
        hints = find_value_hints("restaurants in new york city", index)
        assert len(hints) == 1
        assert "New York" in hints[0]

    def test_short_tokens_ignored(self):
        index = build_value_index(_schema(
            "orders",
            [FieldInfo(name="code", path="code", types=["string"], frequency=1.0,
                       sample_values=["US"])],
        ))
        hints = find_value_hints("orders from us", index)
        assert hints == []

    def test_max_hints_cap(self):
        fields = [
            FieldInfo(name=f"f{i}", path=f"f{i}", types=["string"], frequency=1.0,
                      sample_values=[f"VALUE{i}"])
            for i in range(10)
        ]
        index = build_value_index(_schema("things", fields))
        question = " ".join(f"value{i}" for i in range(10))
        hints = find_value_hints(question, index, max_hints=3)
        assert len(hints) == 3

    def test_same_field_hinted_once_per_call(self):
        index = build_value_index(_schema(
            "orders",
            [FieldInfo(name="status", path="status", types=["string"], frequency=1.0,
                       sample_values=["ACTIVE", "INACTIVE"])],
        ))
        hints = find_value_hints("active or inactive orders?", index)
        # Both tokens resolve to the same (collection, field) — one hint, not two.
        assert len(hints) == 1
