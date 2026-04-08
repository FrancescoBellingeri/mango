# LLM Providers

Mango supports multiple LLM providers out of the box. Each is installed as an optional extra so you only pull in what you need.

---

## Anthropic (Claude)

```bash
pip install mango-ai[anthropic]
```

```python
from mango.integrations.anthropic import AnthropicLlmService

llm = AnthropicLlmService(
    model="claude-sonnet-4-6",      # default
    api_key="sk-ant-...",           # or set ANTHROPIC_API_KEY env var
    max_tokens=4096,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `claude-sonnet-4-6` | Model ID |
| `api_key` | `$ANTHROPIC_API_KEY` | API key |
| `max_tokens` | `4096` | Max response tokens |

---

## OpenAI (GPT)

```bash
pip install mango-ai[openai]
```

```python
from mango.integrations.openai import OpenAiLlmService

llm = OpenAiLlmService(
    model="gpt-4o",                 # default
    api_key="sk-...",               # or set OPENAI_API_KEY env var
    max_completion_tokens=4096,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gpt-4o` | Model ID |
| `api_key` | `$OPENAI_API_KEY` | API key |
| `max_completion_tokens` | `4096` | Max response tokens |

---

## Google (Gemini)

```bash
pip install mango-ai[gemini]
```

```python
from mango.integrations.google import GeminiLlmService

# Google AI Studio
llm = GeminiLlmService(
    model="gemini-3.1-pro-preview",
    api_key="AIza...",              # or set GEMINI_API_KEY env var
)

# Vertex AI
llm = GeminiLlmService(
    model="gemini-3.1-pro-preview",
    vertexai=True,                  # reads GOOGLE_CLOUD_PROJECT from env
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `gemini-3.1-flash-lite-preview` | Model ID |
| `api_key` | `$GOOGLE_API_KEY` | API key (AI Studio only) |
| `vertexai` | `False` | Use Vertex AI instead of AI Studio |
| `max_output_tokens` | `8192` | Max response tokens |
| `temperature` | `1.0` | Sampling temperature |

---

## Using the factory

If you want to select the provider at runtime (e.g. from a config file or CLI flag):

```python
from mango.llm.factory import build_llm

llm = build_llm(provider="anthropic", model="claude-sonnet-4-6")
llm = build_llm(provider="openai")
llm = build_llm(provider="gemini", api_key="AIza...")
```

---

## Bring your own provider

Implement the `LLMService` abstract class from `mango.llm.base`:

```python
from mango.llm.base import LLMService
from mango.llm.models import LLMResponse, Message, ToolDef

class MyCustomLlm(LLMService):
    def __init__(self, **kwargs) -> None:
        try:
            import myclient
        except ImportError:
            raise ImportError("Run: pip install myclient")
        self._client = myclient.Client(**kwargs)

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str = "",
    ) -> LLMResponse:
        # call your provider and return an LLMResponse
        ...

    def get_model_name(self) -> str:
        return "my-model"
```

!!! warning "Lazy imports"
    Always import the provider SDK inside `__init__`, never at module level. This ensures that installing only one provider (e.g. `[gemini]`) does not fail because of a missing `anthropic` or `openai` package.
