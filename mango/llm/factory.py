"""LLM service factory.

Usage::

    from mango.llm.factory import build_llm, PROVIDERS

    llm = build_llm(provider="anthropic", api_key="...")
    llm = build_llm(provider="openai", model="gpt-5.2", api_key="...")
    llm = build_llm(provider="gemini", api_key="...")
"""

from __future__ import annotations

from mango.llm.base import LLMService

PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "gemini")


def build_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMService:
    """Instantiate an LLMService for the given provider.

    Args:
        provider: One of ``"anthropic"``, ``"openai"``, ``"gemini"``.
        model: Model ID override. Uses the provider's default if omitted.
        api_key: API key override. Falls back to the provider's env variable if omitted.

    Returns:
        A configured LLMService instance.

    Raises:
        ValueError: If ``provider`` is not in :data:`PROVIDERS`.
    """
    if provider == "anthropic":
        from mango.integrations.anthropic import AnthropicLlmService
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return AnthropicLlmService(**kwargs)

    if provider == "openai":
        from mango.integrations.openai import OpenAILlmService
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAILlmService(**kwargs)

    if provider == "gemini":
        from mango.integrations.google import GeminiLlmService
        kwargs = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return GeminiLlmService(**kwargs)

    raise ValueError(f"Unknown provider '{provider}'. Must be one of: {', '.join(PROVIDERS)}")
