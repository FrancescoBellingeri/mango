"""Third-party integrations for Mango.

Each sub-module wraps a concrete external library and exposes it via a
clean, consistent class name. Import from here in user scripts:

    from mango.integrations.google import GeminiLlmService
    from mango.integrations.anthropic import AnthropicLlmService
    from mango.integrations.openai import OpenAiLlmService
    from mango.integrations.ollama import OllamaLlmService
    from mango.integrations.mongodb import MongoBackend
    from mango.integrations.chromadb import ChromaAgentMemory
"""

from .mongodb import MongoRunner
from .chromadb import ChromaAgentMemory
from .google import GeminiLlmService
from .anthropic import AnthropicLlmService
from .openai import OpenAiLlmService
from .ollama import OllamaLlmService

__all__ = [
    "MongoRunner",
    "ChromaAgentMemory",
    "GeminiLlmService",
    "AnthropicLlmService",
    "OpenAiLlmService",
    "OllamaLlmService",
]