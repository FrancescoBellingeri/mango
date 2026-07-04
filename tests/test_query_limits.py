"""Query safety-limit tests: server-side time budget (maxTimeMS) and distinct cap.

3b: every query type must carry a server-side time limit so a pathological
    scan cannot hang the worker indefinitely.
3c: `distinct` on a high-cardinality field must be bounded — an unbounded
    distinct can build a huge array and hit the 16 MB BSON limit.

These assert the fixed behaviour; on the pre-fix code they fail.
"""

from __future__ import annotations

import pytest

from mango.core.types import QueryRequest
from mango.tools.mongo_tools import RunMQLTool


@pytest.fixture
def big_backend(mongo_backend):
    """Standard backend plus a collection with 250 distinct 'code' values."""
    mongo_backend._database["bigcol"].insert_many(
        [{"n": i, "code": f"c{i}"} for i in range(250)]
    )
    return mongo_backend


# ---------------------------------------------------------------------------
# 3b — maxTimeMS
# ---------------------------------------------------------------------------


class TestMaxTimeMsConfigured:
    def test_backend_has_configurable_time_budget(self, mongo_backend):
        assert getattr(mongo_backend, "_max_time_ms", None)  # set and non-zero

    def test_aggregate_forwards_max_time_ms(self, big_backend, mocker):
        spy = mocker.spy(type(big_backend._database["bigcol"]), "aggregate")
        req = QueryRequest(operation="aggregate", collection="bigcol",
                           pipeline=[{"$match": {}}], limit=100)
        big_backend.execute_query(req)
        assert spy.call_args.kwargs.get("maxTimeMS") == big_backend._max_time_ms

    def test_count_forwards_max_time_ms(self, big_backend, mocker):
        spy = mocker.spy(type(big_backend._database["bigcol"]), "count_documents")
        big_backend.execute_query(QueryRequest(operation="count", collection="bigcol"))
        assert spy.call_args.kwargs.get("maxTimeMS") == big_backend._max_time_ms

    def test_find_forwards_max_time_ms(self, big_backend, mocker):
        spy = mocker.spy(type(big_backend._database["bigcol"]), "find")
        big_backend.execute_query(
            QueryRequest(operation="find", collection="bigcol", limit=100)
        )
        # max_time_ms is applied to the cursor; it is passed to find() as a kwarg.
        assert spy.call_args.kwargs.get("max_time_ms") == big_backend._max_time_ms


# ---------------------------------------------------------------------------
# 3c — distinct cap
# ---------------------------------------------------------------------------


class TestDistinctCapped:
    async def test_distinct_high_cardinality_is_capped(self, big_backend):
        tool = RunMQLTool(big_backend, max_rows=100)
        result = await tool.execute(
            operation="distinct", collection="bigcol", distinct_field="code"
        )
        assert result.success is True
        assert result.data["row_count"] <= 100

    async def test_distinct_values_correct_under_cap(self, big_backend):
        # Small cardinality: results must still be complete and correct.
        big_backend._database["colors"].insert_many(
            [{"c": "red"}, {"c": "red"}, {"c": "blue"}, {"c": "green"}]
        )
        tool = RunMQLTool(big_backend, max_rows=100)
        result = await tool.execute(
            operation="distinct", collection="colors", distinct_field="c"
        )
        assert result.success is True
        values = {row["c"] for row in result.data["rows"]}
        assert values == {"red", "blue", "green"}
