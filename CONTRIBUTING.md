# Contributing to Mango

Thank you for your interest in contributing to Mango! This guide will help you get started with contributing to the codebase.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)
- [Adding New Features](#adding-new-features)

---

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- A GitHub account
- A MongoDB instance (local or Atlas) for integration tests

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mango-ai.git
   cd mango-ai
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/francescobellingeri/mango-ai.git
   ```

---

## Development Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install the package in editable mode with all LLM providers
pip install -e ".[all]"

# Install development tools
pip install pytest pytest-asyncio
```

### 3. Verify Installation

```bash
# Run unit tests
pytest tests/

# Run a quick smoke test
python -c "from mango import MangoAgent; print('OK')"
```

---

## Code Standards

### Style Guidelines

- Follow PEP 8
- Use descriptive variable and function names
- Add docstrings to all public classes and methods
- Use type hints on all function signatures

**Example:**

```python
"""Module docstring explaining the purpose."""

from __future__ import annotations


class MyService:
    """Short description of what this class does."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import mypackage
        except ImportError:
            raise ImportError(
                "mypackage is not installed. Run: pip install mango-ai[myprovider]"
            )
        self._client = mypackage.Client(api_key=api_key)

    async def do_something(self, query: str) -> str:
        """Short description.

        Args:
            query: The input query.

        Returns:
            The result string.
        """
        ...
```

### Critical Rule: Lazy Imports

SDK packages (`anthropic`, `openai`, `google-genai`, etc.) must **never** be imported at module level. Always import them inside `__init__` so that installing only one provider does not break the others.

```python
# WRONG — breaks users who haven't installed anthropic
import anthropic

class AnthropicLlmService(LLMService):
    ...

# CORRECT
class AnthropicLlmService(LLMService):
    def __init__(self, ...) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install mango-ai[anthropic]")
        self._client = anthropic.Anthropic(...)
```

---

## Testing

### Test Organization

Tests are organized in the `tests/` directory:

- `test_tool_registry.py` — tool registration and execution
- `test_mongo_tools.py` — MongoDB tool behaviour
- `test_mongo_backend.py` — MongoRunner integration
- `test_context_enhancer.py` — agent memory and context injection
- `test_prompt_builder.py` — system prompt generation
- `tests/integration/` — end-to-end tests (require a live MongoDB)

### Running Tests

```bash
# Run all unit tests
pytest tests/

# Run a specific file
pytest tests/test_tool_registry.py -v

# Run integration tests (requires MONGODB_URI env variable)
pytest tests/integration/ -v
```

### Writing Tests

1. **Unit tests** must not require external services (no real MongoDB, no LLM API calls)
2. Use `async def test_*` for all async test functions — pytest-asyncio is configured with `asyncio_mode = "auto"`
3. Mock LLM services with a simple stub:

```python
from mango.llm.base import LLMService, LLMResponse

class MockLlmService(LLMService):
    def chat(self, messages, tools, system_prompt="") -> LLMResponse:
        return LLMResponse(text="mocked answer", tool_calls=[], model="mock", input_tokens=0, output_tokens=0)

    def get_model_name(self) -> str:
        return "mock"
```

4. Use `EphemeralClient` for ChromaDB in tests — never a persistent client
5. Test both success and failure paths
6. Use descriptive test names: `test_<what>_<condition>_<expected>`

---

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feat/my-new-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write your code following the code standards above
- Add tests for any new behaviour
- Update docstrings and examples as needed

### 3. Run All Checks

```bash
# Run the full test suite
pytest tests/

# Quick import smoke test
python -c "from mango import MangoAgent; from mango.tools import ToolRegistry"
```

### 4. Commit Your Changes

Use conventional commit messages:

```bash
git commit -m "feat: add Bedrock LLM provider

- Implements BedrockLlmService with lazy boto3 import
- Adds bedrock extra in pyproject.toml
- Adds unit tests"
```

Prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`

### 5. Push and Open a PR

```bash
git push origin feat/my-new-feature
```

Open a pull request on GitHub with:
- A clear title describing the change
- What was changed and why
- Link to any related issues

---

## Architecture Overview

### Core Components

#### 1. **Agent** (`mango/agent/agent.py`)
The main orchestrator. Runs the tool-calling loop, prunes conversation history, and auto-saves successful interactions to memory.

#### 2. **Tools** (`mango/tools/`)
Modular capabilities the agent can invoke. Each tool extends `Tool` and implements an `async execute(**kwargs)` method.

#### 3. **Tool Registry** (`mango/tools/base.py`)
Manages tool registration and dispatches `execute()` calls by tool name.

#### 4. **Memory** (`mango/integrations/chromadb.py`)
Stores and retrieves past tool uses and free-text notes via ChromaDB vector search.

#### 5. **LLM Services** (`mango/integrations/`)
Pluggable adapters for Anthropic, OpenAI, and Gemini. All share the `LLMService` interface from `mango/llm/base.py`.

#### 6. **MongoDB Runner** (`mango/integrations/mongodb.py`)
Executes MQL queries and introspects collection schemas.

#### 7. **Servers** (`mango/servers/`)
FastAPI SSE server and CLI — both thin wrappers around `MangoAgent`.

### Data Flow

```
User question
     │
     ▼
MangoAgent.ask()
     │
     ├─ retrieve memory (ChromaDB)
     │
     ├─ build system prompt
     │
     └─ LLM loop ──► tool call ──► ToolRegistry.execute()
                                         │
                                         ▼
                                   MongoRunner / ChromaDB
                                         │
                                         ▼
                              result fed back to LLM
                                         │
                                         ▼
                                  final text answer
                                         │
                                         ▼
                              auto-save to memory
```

---

## Adding New Features

### Adding a New LLM Provider

1. Create `mango/integrations/myprovider.py`
2. Implement `LLMService` from `mango.llm.base`
3. Use lazy imports inside `__init__` (see rule above)
4. Register it in `mango/llm/factory.py`
5. Add an optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
myprovider = ["mypackage"]
```

**Minimal example:**

```python
from mango.llm import LLMResponse, LLMService, Message, ToolDef, ToolCall

class MyProviderLlmService(LLMService):
    def __init__(self, api_key: str | None = None, model: str = "default") -> None:
        try:
            import mypackage
        except ImportError:
            raise ImportError("Run: pip install mango-ai[myprovider]")
        self._client = mypackage.Client(api_key=api_key)
        self._model = model

    def chat(self, messages: list[Message], tools: list[ToolDef], system_prompt: str = "") -> LLMResponse:
        ...

    def get_model_name(self) -> str:
        return self._model
```

### Adding a New Tool

1. Create a class that extends `Tool` from `mango.tools.base`
2. Implement `definition` (returns a `ToolDef`) and `async execute(**kwargs) -> ToolResult`
3. Export it from `mango/tools/__init__.py`

```python
from mango.tools.base import Tool, ToolResult
from mango.llm.models import ToolDef

class MyTool(Tool):
    """Does something useful."""

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="my_tool",
            description="Does something useful",
            params=[],
        )

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="done")
```

### Adding a New Memory Backend

1. Implement `MemoryService` from `mango.memory.base`
2. Place it in `mango/integrations/mystore.py`
3. Implement `store()`, `retrieve()`, `save_text()`, `search_text()`, `delete()`, `count()`

---

## Getting Help

- **Issues**: https://github.com/francescobellingeri/mango-ai/issues
- **Discussions**: https://github.com/francescobellingeri/mango-ai/discussions

---

## License

By contributing to Mango, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for contributing to Mango! 🥭
