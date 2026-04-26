# 🥭 Mango — MongoDB AI Agent

**Natural language → MQL → Answers.** The open-source AI agent for MongoDB.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/mango-ai.svg)](https://pypi.org/project/mango-ai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mango.francescobellingeri.com-blue)](https://mango.francescobellingeri.com/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FrancescoBellingeri/mango-ai/blob/main/notebooks/mango_quickstart.ipynb)

> Ask your MongoDB database anything in plain language. Mango translates your question into an MQL query, runs it, and gives you a clear answer — with memory that improves over time.

![Mango demo](src/readme_mango.gif)

<h2 align="center"><a href="https://mango.francescobellingeri.com/">📖 Full documentation</a></h2>

## Install

```bash
pip install mango-ai[anthropic]   # Claude
pip install mango-ai[openai]      # GPT
pip install mango-ai[gemini]      # Gemini
pip install mango-ai[ollama]      # Ollama
pip install mango-ai[all]         # all providers
```

## Quickstart

```python
from mango import MangoAgent
from mango.tools import (
    ToolRegistry,
    ListCollectionsTool,
    SearchCollectionsTool,
    DescribeCollectionTool,
    CollectionStatsTool,
    RunMQLTool,
    SearchSavedCorrectToolUsesTool,
    SaveTextMemoryTool,
    DeleteLastMemoryEntryTool,
)
from mango.servers.fastapi import MangoFastAPIServer
from mango.integrations.anthropic import AnthropicLlmService
from mango.integrations.mongodb import MongoRunner
from mango.integrations.chromadb import ChromaAgentMemory

# Configure your LLM
llm = AnthropicLlmService(
    model="claude-sonnet-4-6",
    api_key="YOUR_API_KEY",
)

# Configure your database
db = MongoRunner()
db.connect("mongodb://localhost:27017/mydb")

# Configure your agent memory
agent_memory = ChromaAgentMemory(
    persist_dir="./chroma_db",
)

# Register tools
tools = ToolRegistry()
tools.register(ListCollectionsTool(db))
tools.register(SearchCollectionsTool(db))
tools.register(DescribeCollectionTool(db))
tools.register(CollectionStatsTool(db))
tools.register(RunMQLTool(db))
tools.register(SearchSavedCorrectToolUsesTool(agent_memory))
tools.register(SaveTextMemoryTool(agent_memory))

# Create your agent
agent = MangoAgent(
    llm_service=llm,
    tool_registry=tools,
    db=db,
    agent_memory=agent_memory,
    introspect=False
)

# Wire up the delete tool AFTER agent creation so it can reference agent state
tools.register(DeleteLastMemoryEntryTool(agent_memory, lambda: agent._last_memory_entry_id))

# Run the server
server = MangoFastAPIServer(agent)
server.run()  # http://localhost:8000
```

Your endpoint is live at `POST /api/v1/ask/stream` — ready to connect to any frontend.

---

## How It Works

```
User question
      │
      ▼
┌─────────────────────────────────────────────┐
│               MANGO AGENT                    │
│                                              │
│  1. Inject training examples (gold-standard) │
│  2. Search memory for similar past queries   │
│  3. Build system prompt with schema context  │
│  4. LLM decides which tools to call          │
│  5. Validate MQL before execution            │
│  6. Execute tools against MongoDB            │
│  7. Auto-retry on fixable errors (max 2x)    │
│  8. Stream natural language answer           │
│  9. Auto-save successful queries to memory   │
└─────────────────────────────────────────────┘
      │
      ▼
SSE stream → your frontend
```

**The learning loop:** steps 1, 2, and 9 make Mango smarter over time. Training examples are injected first — the LLM uses them directly without exploring the schema, which cuts latency and improves accuracy. Auto-saved examples accumulate during use. A novel question triggers full reasoning; a familiar one is answered in one shot.

---

## Why Mango?

Vanna.ai has solved text-to-SQL elegantly. MongoDB has no equivalent. The challenges are fundamentally different:

- **No explicit schema** — collections have no DDL. Mango infers schema by sampling documents.
- **Nested documents** — queries must navigate arrays, subdocuments, and dotted paths.
- **Aggregation pipelines** — complex analytics require multi-stage JSON pipelines, not flat SQL strings.
- **No JOINs** — relationships are handled via `$lookup` or application-level references.

Mango is the first production-grade framework for natural language interaction with MongoDB.

---

## SSE Streaming API

Every question is streamed via Server-Sent Events. Each `data:` line is a JSON event:

```
POST /api/v1/ask/stream
{"question": "How many orders were placed last week?"}
```

```
data: {"type": "session",     "session_id": "abc123"}
data: {"type": "tool_call",   "tool_name": "list_collections", "tool_args": {}}
data: {"type": "tool_result", "tool_name": "list_collections", "success": true, "preview": "orders, customers, products..."}
data: {"type": "tool_call",   "tool_name": "run_mql", "tool_args": {"operation": "aggregate", "collection": "orders", ...}}
data: {"type": "tool_result", "tool_name": "run_mql", "success": true, "preview": "[{\"total\": 1247}]"}
data: {"type": "answer",      "text": "1,247 orders were placed in the last 7 days."}
data: {"type": "done",        "iterations": 2, "input_tokens": 1820, "output_tokens": 94}
```

Multi-turn conversations are supported — pass the same `session_id` to continue a thread.

---

## Agent Memory

Mango learns from every successful interaction using ChromaDB as a vector store. Memory is organized in three layers:

| Layer | How it's populated | Priority |
|-------|--------------------|----------|
| **Training** | You load verified examples explicitly | Injected first — LLM uses directly |
| **Auto-save** | Populated automatically during use | Injected as few-shot context |
| **Text notes** | You add domain knowledge manually | Retrieved when relevant |

**Load training data** (bulk, from a JSONL file):

```bash
mango train --file knowledge.jsonl
```

**Pre-load domain knowledge:**

```python
await memory.save_text("'active customer' means placed an order in the last 90 days")
await memory.save_text("'revenue' always refers to the total_amount field in orders")
await memory.save_text("the 'status' field uses: 1=pending, 2=shipped, 3=delivered, 4=cancelled")
```

**Correct mistakes in chat** — no UI needed:

```
User: that query was wrong, delete it
Mango: Done — removed from memory ✓
```

→ Requires `DeleteLastMemoryEntryTool` registered after agent creation (see Quickstart).

---

## Available Tools

| Tool | Description |
|------|-------------|
| `list_collections` | List all collections. Grouped view for large databases (100+ collections). |
| `search_collections` | Search collections by name pattern (supports glob: `order*`, `*_log`). |
| `describe_collection` | Full schema for a collection: field types, frequencies, indexes, references. |
| `collection_stats` | Document count and storage size for a collection. |
| `run_mql` | Execute a read-only MongoDB query: `find`, `aggregate`, `count`, `distinct`. Includes automatic MQL validation before execution. |
| `search_saved_correct_tool_uses` | Search memory for similar past interactions. |
| `save_text_memory` | Save free-form knowledge about the database for future queries. |
| `explain_query` | *(opt-in)* Explain a query step-by-step in plain language + MongoDB execution stats. |
| `delete_last_memory_entry` | *(opt-in)* Remove the last auto-saved entry when the user says a result was wrong. |

> **Read-only by design.** `run_mql` only accepts `find`, `aggregate`, `count`, `distinct`. Write operations are rejected at the tool level.

> **MQL validation.** Every `run_mql` call is validated before hitting the database — collection names, field names, and operators are checked against the live schema. Errors come back with hints (`did you mean 'order_total'?`) so the LLM can self-correct.

---

## Pluggable Architecture

Mango is built on abstract interfaces — swap any component without touching your agent code.

### LLM Providers

```python
from mango.integrations.anthropic import AnthropicLlmService
from mango.integrations.openai import OpenAILlmService
from mango.integrations.google import GeminiLlmService
from mango.integrations.ollama import OllamaLlmService

llm = AnthropicLlmService(model="claude-sonnet-4-6")
llm = OpenAILlmService(model="gpt-5.4")
llm = GeminiLlmService(model="gemini-3.1-pro-preview")
llm = OllamaLlmService(model="qwen3.5:9b")  # fully local, no API key needed
```

### Custom Tools

```python
from mango.tools.base import Tool, ToolResult
from mango.llm import ToolDef, ToolParam

class MyCustomTool(Tool):
    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="my_tool",
            description="Does something useful",
            params=[ToolParam(name="input", type="string", description="...")]
        )

    async def execute(self, **kwargs) -> ToolResult:
        result = do_something(kwargs["input"])
        return ToolResult(success=True, data=result)

tools.register(MyCustomTool())
```

### Memory Backends

```python
# Default: ChromaDB (local, no infrastructure needed)
from mango.integrations.chromadb import ChromaAgentMemory
memory = ChromaAgentMemory(persist_dir="./mango_memory")

# Implement your own: just inherit MemoryService
class MyPineconeMemory(MemoryService):
    async def store(self, entry: MemoryEntry) -> None: ...
    async def retrieve(self, question: str, top_k: int) -> list[MemoryEntry]: ...
    async def save_text(self, content: str) -> str: ...
```

---

## Large Databases

Mango handles databases with hundreds or thousands of collections without token explosion.

- **Adaptive collection listing** — databases with 100+ collections are automatically grouped by name pattern (`contest_*`, `user_*`) instead of listing every single one
- **On-demand schema** — schema details are only fetched when the LLM needs them, not injected upfront
- **Turn-based conversation pruning** — conversation history is automatically trimmed to keep token usage stable across long sessions
- **Auto-save to memory** — schema discoveries are persisted so the LLM doesn't re-introspect the same collections repeatedly

---

## Multi-turn Conversations

Mango maintains conversation history across questions in the same session:

```
User: How many orders were placed last week?
Mango: 1,247 orders were placed in the last 7 days.

User: And how many of those were delivered?
Mango: Of last week's 1,247 orders, 891 (71%) have been delivered.

User: Which customer placed the most?
Mango: Alice Johnson (customer_id: 64a3f...) placed 8 orders last week.
```

Follow-up questions work naturally — no need to repeat context.

---

## Technology Stack

| Component | Library |
|-----------|---------|
| MongoDB driver | pymongo 4.x |
| LLM (primary) | anthropic (Claude) |
| LLM (secondary) | openai (GPT-4), google-generativeai (Gemini), ollama (local) |
| Vector store | chromadb 1.x |
| Server | FastAPI + uvicorn |
| CLI | rich + prompt_toolkit |
| Data | pandas |
| Testing | pytest + mongomock |

---

## Roadmap

**Foundation**
- [x] MongoDB backend with full schema introspection
- [x] Pluggable LLM providers (Anthropic, OpenAI, Gemini)
- [x] `RunMQLTool` — read-only find, aggregate, count, distinct
- [x] Streaming FastAPI server (SSE)
- [x] CLI interface

**Memory & Learning**
- [x] ChromaDB agent memory with auto-save
- [x] `SaveTextMemoryTool` — business glossary and domain knowledge
- [x] Adaptive collection grouping for large databases
- [x] Multi-turn conversation with automatic pruning
- [x] Memory export/import (JSON)
- [x] `mango train` — bulk training data pre-loading from JSONL
- [x] `DeleteLastMemoryEntryTool` — in-chat memory correction

**Polish**
- [x] `MQLValidator` — pre-execution validation with field/operator hints
- [x] `ExplainQueryTool` — pipeline explanation in plain language + execution stats
- [x] Retry-with-error flow (max 2 retries on fixable query errors)
- [x] Ollama integration — fully local inference, no API key
- [ ] `VisualizeDataTool` — charts and tables in CLI

**Expansion**
- [ ] Atlas Vector Search memory backend
- [ ] Redis backend (experimental)
- [ ] Cassandra backend (experimental)
- [ ] DynamoDB backend (experimental)
- [ ] Memory analytics (most common queries, accuracy tracking)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code standards, and PR process.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with ❤️ by Francesco Bellingeri** | Inspired by [Vanna.ai](https://vanna.ai)
