"""
End-to-end workflow tests — real LLM providers + mongomock database.

Each test sends a natural language question to MangoAgent and asserts that
the answer contains the expected information. The database holds three users
(Alice/30, Bob/25, Charlie/35) and three orders (Widget×5, Gadget×2, Widget×10).

Mirrors Vanna's test_agents.py approach:
  - One test class per provider, marked with the corresponding API-key marker
  - Tests auto-skip when the required key is absent
  - A shared helper (assert_workflow) runs the same questions for every provider

Run only one provider:
    pytest tests/integration/ -m openai -v
    pytest tests/integration/ -m anthropic -v
    pytest tests/integration/ -m gemini -v

Run all integration tests:
    pytest tests/integration/ -v
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import TEST_DB_FACTS, make_agent


# ---------------------------------------------------------------------------
# Shared workflow assertions
# ---------------------------------------------------------------------------


async def assert_answer_contains(agent, question: str, *expected_substrings: str) -> str:
    """Ask the agent a question and verify each expected substring is in the answer.

    Returns the full answer text for additional assertions in the calling test.
    """
    resp = await agent.ask(question)
    answer = resp.answer
    assert answer, f"Agent returned an empty answer for: {question!r}"
    for expected in expected_substrings:
        assert expected in answer, (
            f"Expected {expected!r} in answer for question {question!r}.\n"
            f"Got: {answer}"
        )
    return answer


async def run_standard_workflow(agent) -> None:
    """Run the canonical set of questions against any agent.

    These cover the full tool pipeline:
      - list_collections → verify database is reachable
      - run_mql count   → verify numeric results
      - run_mql distinct → verify string results
    """
    # --- Q1: How many users? (run_mql count) ---
    answer = await assert_answer_contains(
        agent,
        "How many users are there in the database?",
        str(TEST_DB_FACTS["user_count"]),   # "3"
    )
    assert len(answer) > 0

    # --- Q2: What products are in the orders? (run_mql distinct / find) ---
    # Start a fresh conversation turn so the agent re-runs tools.
    agent.reset_conversation()
    await assert_answer_contains(
        agent,
        "What products appear in the orders collection?",
        "Widget",
        "Gadget",
    )

    # --- Q3: Total qty of widgets (run_mql aggregate) ---
    agent.reset_conversation()
    await assert_answer_contains(
        agent,
        "What is the total quantity of Widget items across all orders?",
        str(TEST_DB_FACTS["widget_total_qty"]),  # "15"
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.openai
@pytest.mark.integration
class TestOpenAIWorkflow:
    """End-to-end workflow using the OpenAI provider."""

    async def test_user_count(self, openai_llm, integration_backend, integration_registry):
        agent = make_agent(openai_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "How many users are there?",
            str(TEST_DB_FACTS["user_count"]),
        )

    async def test_distinct_products(
        self, openai_llm, integration_backend, integration_registry
    ):
        agent = make_agent(openai_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "List the distinct products in the orders collection.",
            "Widget",
            "Gadget",
        )

    async def test_aggregate_qty(
        self, openai_llm, integration_backend, integration_registry
    ):
        agent = make_agent(openai_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "What is the total quantity of Widget products ordered?",
            str(TEST_DB_FACTS["widget_total_qty"]),
        )

    async def test_tool_calls_are_made(
        self, openai_llm, integration_backend, integration_registry
    ):
        """The agent must call at least one tool to answer a data question."""
        agent = make_agent(openai_llm, integration_backend, integration_registry)
        resp = await agent.ask("How many users are registered?")
        assert len(resp.tool_calls_made) >= 1, (
            "Agent should call at least one tool to answer a database question"
        )

    async def test_run_mql_is_used(
        self, openai_llm, integration_backend, integration_registry
    ):
        """run_mql must appear in the tool calls when counting documents."""
        agent = make_agent(openai_llm, integration_backend, integration_registry)
        resp = await agent.ask("Count the total number of users.")
        assert "run_mql" in resp.tool_calls_made, (
            f"Expected run_mql in tool calls, got: {resp.tool_calls_made}"
        )

    async def test_full_workflow(
        self, openai_llm, integration_backend, integration_registry
    ):
        """Run all standard questions in sequence."""
        agent = make_agent(openai_llm, integration_backend, integration_registry)
        await run_standard_workflow(agent)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.anthropic
@pytest.mark.integration
class TestAnthropicWorkflow:
    """End-to-end workflow using the Anthropic provider."""

    async def test_user_count(
        self, anthropic_llm, integration_backend, integration_registry
    ):
        agent = make_agent(anthropic_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "How many users are there?",
            str(TEST_DB_FACTS["user_count"]),
        )

    async def test_distinct_products(
        self, anthropic_llm, integration_backend, integration_registry
    ):
        agent = make_agent(anthropic_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "List the distinct products in the orders collection.",
            "Widget",
            "Gadget",
        )

    async def test_aggregate_qty(
        self, anthropic_llm, integration_backend, integration_registry
    ):
        agent = make_agent(anthropic_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "What is the total quantity of Widget products ordered?",
            str(TEST_DB_FACTS["widget_total_qty"]),
        )

    async def test_tool_calls_are_made(
        self, anthropic_llm, integration_backend, integration_registry
    ):
        agent = make_agent(anthropic_llm, integration_backend, integration_registry)
        resp = await agent.ask("How many users are registered?")
        assert len(resp.tool_calls_made) >= 1

    async def test_run_mql_is_used(
        self, anthropic_llm, integration_backend, integration_registry
    ):
        agent = make_agent(anthropic_llm, integration_backend, integration_registry)
        resp = await agent.ask("Count the total number of users.")
        assert "run_mql" in resp.tool_calls_made

    async def test_full_workflow(
        self, anthropic_llm, integration_backend, integration_registry
    ):
        agent = make_agent(anthropic_llm, integration_backend, integration_registry)
        await run_standard_workflow(agent)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@pytest.mark.gemini
@pytest.mark.integration
class TestGeminiWorkflow:
    """End-to-end workflow using the Gemini provider."""

    async def test_user_count(self, gemini_llm, integration_backend, integration_registry):
        agent = make_agent(gemini_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "How many users are there?",
            str(TEST_DB_FACTS["user_count"]),
        )

    async def test_distinct_products(
        self, gemini_llm, integration_backend, integration_registry
    ):
        agent = make_agent(gemini_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "List the distinct products in the orders collection.",
            "Widget",
            "Gadget",
        )

    async def test_aggregate_qty(
        self, gemini_llm, integration_backend, integration_registry
    ):
        agent = make_agent(gemini_llm, integration_backend, integration_registry)
        await assert_answer_contains(
            agent,
            "What is the total quantity of Widget products ordered?",
            str(TEST_DB_FACTS["widget_total_qty"]),
        )

    async def test_tool_calls_are_made(
        self, gemini_llm, integration_backend, integration_registry
    ):
        agent = make_agent(gemini_llm, integration_backend, integration_registry)
        resp = await agent.ask("How many users are registered?")
        assert len(resp.tool_calls_made) >= 1

    async def test_run_mql_is_used(
        self, gemini_llm, integration_backend, integration_registry
    ):
        agent = make_agent(gemini_llm, integration_backend, integration_registry)
        resp = await agent.ask("Count the total number of users.")
        assert "run_mql" in resp.tool_calls_made

    async def test_full_workflow(
        self, gemini_llm, integration_backend, integration_registry
    ):
        agent = make_agent(gemini_llm, integration_backend, integration_registry)
        await run_standard_workflow(agent)
