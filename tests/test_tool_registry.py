"""Tests for mango.tools.base — ToolResult, Tool ABC, ToolRegistry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mango.llm.models import ToolDef
from mango.tools.base import Tool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Minimal concrete Tool for testing
# ---------------------------------------------------------------------------


class _EchoTool(Tool):
    """Returns its input arguments as the result."""

    @property
    def definition(self) -> ToolDef:
        return ToolDef(name="echo", description="Echo args back.", params=[])

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=kwargs)


class _FailingTool(Tool):
    """Always raises an exception when executed."""

    @property
    def definition(self) -> ToolDef:
        return ToolDef(name="fail", description="Always fails.", params=[])

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_string_data(self):
        result = ToolResult(success=True, data="hello world")
        assert result.as_text() == "hello world"

    def test_success_dict_data_serialised_to_json(self):
        result = ToolResult(success=True, data={"count": 42})
        text = result.as_text()
        parsed = json.loads(text)
        assert parsed["count"] == 42

    def test_success_list_data(self):
        result = ToolResult(success=True, data=[1, 2, 3])
        text = result.as_text()
        assert json.loads(text) == [1, 2, 3]

    def test_failure_returns_error_prefix(self):
        result = ToolResult(success=False, error="something went wrong")
        assert result.as_text() == "ERROR: something went wrong"

    def test_success_none_data_returns_null_json(self):
        result = ToolResult(success=True, data=None)
        assert result.as_text() == "null"

    def test_unserializable_data_serialised_via_default_str(self):
        # json.dumps(default=str) calls str() on unknown objects and wraps the
        # result as a JSON string — so the output includes surrounding quotes.
        class Unserializable:
            def __str__(self):
                return "custom-repr"

        result = ToolResult(success=True, data=Unserializable())
        import json
        assert result.as_text() == json.dumps("custom-repr")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_len(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        assert len(registry) == 1

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        assert "echo" in registry
        assert "nonexistent" not in registry

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_EchoTool())

    def test_get_definitions_returns_list(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].name == "echo"

    async def test_execute_known_tool(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        result = await registry.execute("echo", message="hi")
        assert result.success is True
        assert result.data == {"message": "hi"}

    async def test_execute_unknown_tool_returns_failed_result(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent_tool")
        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_execute_exception_in_tool_returns_failed_result(self):
        registry = ToolRegistry()
        registry.register(_FailingTool())
        result = await registry.execute("fail")
        assert result.success is False
        assert "intentional failure" in result.error

    def test_empty_registry(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.get_definitions() == []
