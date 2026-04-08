"""Abstract base class for LLM service integrations.

Every LLM provider (Anthropic, OpenAI, ...) must implement this interface.
The agent always talks to an LLMService — never to a provider SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class LLMService(ABC):
    """Abstract interface for LLM provider integrations.

    Implementations must support tool/function calling, as Mango relies on
    it to dispatch queries to the correct tool instead of generating raw text.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str = "",
    ) -> LLMResponse:
        """Send a conversation turn to the LLM and return its response.

        Args:
            messages: Conversation history including the latest user message.
            tools: Tool definitions available to the LLM this turn.
            system_prompt: System prompt injected before the conversation.

        Returns:
            LLMResponse with optional text and/or tool call requests.

        Raises:
            LLMError: If the provider returns an error.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier used by this service.

        Returns:
            Model name string, e.g. 'claude-sonnet-4-6'.
        """
