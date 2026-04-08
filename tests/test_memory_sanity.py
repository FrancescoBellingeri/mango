"""
Sanity tests for Mango interfaces.

These tests verify structure and contracts without making any API calls:
  - MemoryService ABC defines the correct abstract methods
  - MemoryEntry has all required fields
  - ChromaMemoryService implements every abstract method
  - LLMService ABC defines the correct abstract methods
  - Each LLM module exports the required class
  - build_llm() factory accepts all declared providers

No external services are required.
"""

from __future__ import annotations

from abc import ABC
from inspect import signature


# ---------------------------------------------------------------------------
# MemoryService interface
# ---------------------------------------------------------------------------


class TestMemoryServiceInterface:
    """The MemoryService ABC must expose a stable contract."""

    def test_is_abstract(self):
        from mango.memory.base import MemoryService
        assert issubclass(MemoryService, ABC)

    def test_cannot_instantiate_directly(self):
        import pytest
        from mango.memory.base import MemoryService
        with pytest.raises(TypeError):
            MemoryService()  # type: ignore[abstract]

    def test_has_store_method(self):
        from mango.memory.base import MemoryService
        assert hasattr(MemoryService, "store")
        assert getattr(MemoryService.store, "__isabstractmethod__", False)

    def test_has_retrieve_method(self):
        from mango.memory.base import MemoryService
        assert hasattr(MemoryService, "retrieve")
        assert getattr(MemoryService.retrieve, "__isabstractmethod__", False)

    def test_has_delete_method(self):
        from mango.memory.base import MemoryService
        assert hasattr(MemoryService, "delete")
        assert getattr(MemoryService.delete, "__isabstractmethod__", False)

    def test_has_count_method(self):
        from mango.memory.base import MemoryService
        assert hasattr(MemoryService, "count")
        assert getattr(MemoryService.count, "__isabstractmethod__", False)

    def test_has_save_text_method(self):
        from mango.memory.base import MemoryService
        assert hasattr(MemoryService, "save_text")
        assert getattr(MemoryService.save_text, "__isabstractmethod__", False)

    def test_has_search_text_method(self):
        from mango.memory.base import MemoryService
        assert hasattr(MemoryService, "search_text")
        assert getattr(MemoryService.search_text, "__isabstractmethod__", False)

    def test_retrieve_signature(self):
        from mango.memory.base import MemoryService
        sig = signature(MemoryService.retrieve)
        params = list(sig.parameters.keys())
        assert "question" in params
        assert "top_k" in params

    def test_store_signature(self):
        from mango.memory.base import MemoryService
        sig = signature(MemoryService.store)
        params = list(sig.parameters.keys())
        assert "entry" in params


# ---------------------------------------------------------------------------
# MemoryEntry model
# ---------------------------------------------------------------------------


class TestMemoryEntryModel:
    """MemoryEntry is the data contract between memory implementations."""

    def test_import(self):
        from mango.memory.models import MemoryEntry
        assert MemoryEntry is not None

    def test_required_fields_present(self):
        from mango.memory.models import MemoryEntry
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(MemoryEntry)}
        required = {"id", "question", "tool_name", "tool_args", "result_summary"}
        assert required.issubset(field_names)

    def test_similarity_has_default(self):
        from mango.memory.models import MemoryEntry
        entry = MemoryEntry(
            id="x",
            question="q",
            tool_name="run_mql",
            tool_args={},
            result_summary="ok",
        )
        assert entry.similarity == 0.0

    def test_instantiation(self):
        from mango.memory.models import MemoryEntry
        entry = MemoryEntry(
            id="abc-123",
            question="How many orders?",
            tool_name="run_mql",
            tool_args={"operation": "count", "collection": "orders"},
            result_summary="3 orders found.",
            similarity=0.92,
        )
        assert entry.id == "abc-123"
        assert entry.tool_name == "run_mql"
        assert entry.similarity == 0.92


# ---------------------------------------------------------------------------
# ChromaMemoryService implementation
# ---------------------------------------------------------------------------


class TestChromaMemoryServiceSanity:
    """ChromaMemoryService must fully implement MemoryService."""

    def test_import(self):
        from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService
        assert ChromaMemoryService is not None

    def test_implements_memory_service(self):
        from mango.memory.base import MemoryService
        from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService
        assert issubclass(ChromaMemoryService, MemoryService)

    def test_all_abstract_methods_implemented(self):
        from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService
        for method_name in ("store", "retrieve", "delete", "count"):
            method = getattr(ChromaMemoryService, method_name)
            assert not getattr(method, "__isabstractmethod__", False), (
                f"{method_name} should be implemented, not abstract"
            )

    def test_instantiation_ephemeral(self):
        from mango.integrations.chromadb import ChromaAgentMemory as ChromaMemoryService
        svc = ChromaMemoryService(persist_dir=":memory:")
        assert svc is not None

    def test_make_entry_id_is_unique(self):
        from mango.integrations.chromadb import make_entry_id
        ids = {make_entry_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# LLMService interface
# ---------------------------------------------------------------------------


class TestLLMServiceInterface:
    """LLMService ABC must expose the required contract."""

    def test_is_abstract(self):
        from mango.llm.base import LLMService
        assert issubclass(LLMService, ABC)

    def test_cannot_instantiate_directly(self):
        import pytest
        from mango.llm.base import LLMService
        with pytest.raises(TypeError):
            LLMService()  # type: ignore[abstract]

    def test_has_chat_method(self):
        from mango.llm.base import LLMService
        assert hasattr(LLMService, "chat")
        assert getattr(LLMService.chat, "__isabstractmethod__", False)

    def test_has_get_model_name_method(self):
        from mango.llm.base import LLMService
        assert hasattr(LLMService, "get_model_name")
        assert getattr(LLMService.get_model_name, "__isabstractmethod__", False)

    def test_chat_signature(self):
        from mango.llm.base import LLMService
        sig = signature(LLMService.chat)
        params = list(sig.parameters.keys())
        assert "messages" in params
        assert "tools" in params
        assert "system_prompt" in params


# ---------------------------------------------------------------------------
# LLM dataclasses
# ---------------------------------------------------------------------------


class TestLLMDataclasses:
    """ToolCall, LLMResponse, and Message must have required fields."""

    def test_tool_call_fields(self):
        from mango.llm.models import ToolCall
        tc = ToolCall(tool_name="run_mql", tool_args={"operation": "count"})
        assert tc.tool_name == "run_mql"
        assert tc.tool_call_id == ""
        assert tc.thought_signature is None

    def test_llm_response_has_tool_calls(self):
        from mango.llm.models import LLMResponse
        resp = LLMResponse(text="hello", tool_calls=[])
        assert resp.has_tool_calls is False

    def test_tool_def_import(self):
        from mango.llm.models import ToolDef, ToolParam
        td = ToolDef(
            name="list_collections",
            description="List all collections",
            params=[ToolParam(name="x", type="string", description="x")],
        )
        assert td.name == "list_collections"

    def test_message_fields(self):
        from mango.llm.models import Message
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.tool_call_id is None


# ---------------------------------------------------------------------------
# LLM provider modules
# ---------------------------------------------------------------------------


class TestLLMProviderModules:
    """Each provider module must export its service class."""

    def test_openai_service_import(self):
        from mango.integrations.openai import OpenAiLlmService as OpenAIService
        assert OpenAIService is not None

    def test_openai_implements_llm_service(self):
        from mango.llm.base import LLMService
        from mango.integrations.openai import OpenAiLlmService as OpenAIService
        assert issubclass(OpenAIService, LLMService)

    def test_openai_has_default_model(self):
        from mango.integrations.openai import OpenAiLlmService as OpenAIService
        assert hasattr(OpenAIService, "DEFAULT_MODEL")
        assert isinstance(OpenAIService.DEFAULT_MODEL, str)

    def test_anthropic_service_import(self):
        from mango.integrations.anthropic import AnthropicLlmService as AnthropicService
        assert AnthropicService is not None

    def test_anthropic_implements_llm_service(self):
        from mango.llm.base import LLMService
        from mango.integrations.anthropic import AnthropicLlmService as AnthropicService
        assert issubclass(AnthropicService, LLMService)

    def test_gemini_service_import(self):
        from mango.integrations.google import GeminiLlmService as GeminiService
        assert GeminiService is not None

    def test_gemini_implements_llm_service(self):
        from mango.llm.base import LLMService
        from mango.integrations.google import GeminiLlmService as GeminiService
        assert issubclass(GeminiService, LLMService)

    def test_all_providers_have_chat_method(self):
        from mango.integrations.openai import OpenAiLlmService as OpenAIService
        from mango.integrations.anthropic import AnthropicLlmService as AnthropicService
        from mango.integrations.google import GeminiLlmService as GeminiService
        for cls in (OpenAIService, AnthropicService, GeminiService):
            assert hasattr(cls, "chat")
            assert not getattr(cls.chat, "__isabstractmethod__", False), (
                f"{cls.__name__}.chat should be implemented"
            )

    def test_all_providers_have_get_model_name_method(self):
        from mango.integrations.openai import OpenAiLlmService as OpenAIService
        from mango.integrations.anthropic import AnthropicLlmService as AnthropicService
        from mango.integrations.google import GeminiLlmService as GeminiService
        for cls in (OpenAIService, AnthropicService, GeminiService):
            assert hasattr(cls, "get_model_name")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


class TestLLMFactory:
    """build_llm() must accept all declared providers without API calls."""

    def test_factory_import(self):
        from mango.llm.factory import build_llm, PROVIDERS
        assert build_llm is not None
        assert len(PROVIDERS) == 3

    def test_providers_tuple_contents(self):
        from mango.llm.factory import PROVIDERS
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS
        assert "gemini" in PROVIDERS

    def test_unknown_provider_raises(self):
        import pytest
        from mango.llm.factory import build_llm
        with pytest.raises(ValueError, match="Unknown provider"):
            build_llm(provider="cohere")

    def test_openai_returns_service(self):
        from mango.llm.factory import build_llm
        from mango.integrations.openai import OpenAiLlmService as OpenAIService
        # api_key=None is fine — the service stores it, doesn't validate at init.
        svc = build_llm(provider="openai", api_key="test-key")
        assert isinstance(svc, OpenAIService)

    def test_anthropic_returns_service(self):
        from mango.llm.factory import build_llm
        from mango.integrations.anthropic import AnthropicLlmService as AnthropicService
        svc = build_llm(provider="anthropic", api_key="test-key")
        assert isinstance(svc, AnthropicService)

    def test_gemini_returns_service(self):
        from mango.llm.factory import build_llm
        from mango.integrations.google import GeminiLlmService as GeminiService
        svc = build_llm(provider="gemini", api_key="test-key")
        assert isinstance(svc, GeminiService)

    def test_model_override(self):
        from mango.llm.factory import build_llm
        svc = build_llm(provider="openai", model="gpt-4o-mini", api_key="test-key")
        assert svc.get_model_name() == "gpt-4o-mini"
