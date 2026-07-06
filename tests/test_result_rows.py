"""execute_query returns rows (list[dict]), not a lossy DataFrame roundtrip.

The pandas roundtrip (docs → DataFrame → to_json → json.loads) corrupts two
things that matter to the LLM and to result-set scoring:

  * a document missing a field gets that field injected as null (DataFrame
    unifies columns) — "absent" and "null" are distinct in MongoDB;
  * an int column with one missing value becomes float (42 → 42.0).

Returning the stringified documents directly preserves both.
"""

from __future__ import annotations

from mango.core.types import QueryRequest
from mango.tools.mongo_tools import RunMQLTool


# ---------------------------------------------------------------------------
# Backend contract: execute_query returns list[dict]
# ---------------------------------------------------------------------------


class TestExecuteQueryReturnsRows:
    def test_find_returns_list_of_dicts(self, mongo_backend):
        rows = mongo_backend.execute_query(
            QueryRequest(operation="find", collection="users")
        )
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        assert len(rows) == 3

    def test_count_returns_single_row_list(self, mongo_backend):
        rows = mongo_backend.execute_query(
            QueryRequest(operation="count", collection="users")
        )
        assert rows == [{"count": 3}]

    def test_distinct_returns_rows(self, mongo_backend):
        rows = mongo_backend.execute_query(
            QueryRequest(operation="distinct", collection="orders", distinct_field="product")
        )
        assert isinstance(rows, list)
        assert {r["product"] for r in rows} == {"Widget", "Gadget"}

    def test_empty_find_returns_empty_list(self, mongo_backend):
        rows = mongo_backend.execute_query(
            QueryRequest(operation="find", collection="users", filter={"name": "Nobody"})
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Lossy-roundtrip corruption is gone
# ---------------------------------------------------------------------------


class TestAbsentVsNullPreserved:
    async def test_absent_field_not_injected_as_null(self, mongo_backend):
        mongo_backend._database["hetero"].insert_many([{"a": 1, "b": 2}, {"a": 3}])
        tool = RunMQLTool(mongo_backend, max_rows=100)
        res = await tool.execute(operation="find", collection="hetero", projection={"_id": 0})
        by_a = {r["a"]: r for r in res.data["rows"]}
        assert "b" in by_a[1]
        assert "b" not in by_a[3]          # never had 'b' → must stay absent, not null

    async def test_int_field_stays_int(self, mongo_backend):
        mongo_backend._database["hetero2"].insert_many([{"n": 1, "x": 10}, {"n": 2}])
        tool = RunMQLTool(mongo_backend, max_rows=100)
        res = await tool.execute(operation="find", collection="hetero2", projection={"_id": 0})
        by_n = {r["n"]: r for r in res.data["rows"]}
        assert by_n[1]["x"] == 10
        assert isinstance(by_n[1]["x"], int)   # not coerced to 10.0

    async def test_objectid_and_date_are_json_safe(self, mongo_backend):
        # ObjectId → str, datetime → ISO string, so the result serialises cleanly.
        from datetime import datetime
        mongo_backend._database["evts"].insert_many([{"when": datetime(2024, 1, 2, 3, 4, 5)}])
        tool = RunMQLTool(mongo_backend, max_rows=100)
        res = await tool.execute(operation="find", collection="evts")
        row = res.data["rows"][0]
        assert isinstance(row["_id"], str)     # ObjectId stringified
        assert isinstance(row["when"], str)    # datetime → ISO string
        assert row["when"].startswith("2024-01-02")
