"""Result-cap tests for run_mql.

`RunMQLTool` advertises "Results are capped at 100 rows by default", and the
find path enforces it. The aggregate path must enforce the same cap: an
aggregation pipeline without a terminal $limit could otherwise stream an
unbounded number of rows into memory and into the LLM context.

These tests assert the capped behaviour. On the pre-fix code the
`test_aggregate_*` cases fail (aggregate returns everything).
"""

from __future__ import annotations

import pytest

from mango.core.types import QueryRequest
from mango.tools.mongo_tools import RunMQLTool


@pytest.fixture
def big_backend(mongo_backend):
    """The standard in-memory backend plus a 250-doc collection."""
    mongo_backend._database["bigcol"].insert_many(
        [{"n": i, "grp": i % 200} for i in range(250)]
    )
    return mongo_backend


class TestAggregateRespectsCap:
    async def test_aggregate_without_limit_is_capped(self, big_backend):
        tool = RunMQLTool(big_backend, max_rows=100)
        result = await tool.execute(
            operation="aggregate",
            collection="bigcol",
            pipeline=[{"$match": {}}],
        )
        assert result.success is True
        assert result.data["row_count"] <= 100

    async def test_aggregate_group_without_limit_is_capped(self, big_backend):
        # 200 distinct groups → would return 200 rows uncapped.
        tool = RunMQLTool(big_backend, max_rows=100)
        result = await tool.execute(
            operation="aggregate",
            collection="bigcol",
            pipeline=[{"$group": {"_id": "$grp", "c": {"$sum": 1}}}],
        )
        assert result.success is True
        assert result.data["row_count"] <= 100

    async def test_aggregate_honours_smaller_own_limit(self, big_backend):
        # The pipeline's own smaller $limit must win over the cap.
        tool = RunMQLTool(big_backend, max_rows=100)
        result = await tool.execute(
            operation="aggregate",
            collection="bigcol",
            pipeline=[{"$match": {}}, {"$limit": 5}],
        )
        assert result.success is True
        assert result.data["row_count"] == 5

    async def test_backend_aggregate_respects_request_limit(self, big_backend):
        # Direct backend call: req.limit must bound aggregate output.
        req = QueryRequest(
            operation="aggregate",
            collection="bigcol",
            pipeline=[{"$match": {}}],
            limit=100,
        )
        df = big_backend.execute_query(req)
        assert len(df) <= 100


class TestFindStillCapped:
    """Control: the find path was already capped and must stay capped."""

    async def test_find_without_limit_is_capped(self, big_backend):
        tool = RunMQLTool(big_backend, max_rows=100)
        result = await tool.execute(operation="find", collection="bigcol")
        assert result.success is True
        assert result.data["row_count"] <= 100
