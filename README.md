# 🥭 Mango — MongoDB AI Agent

**Natural language → MQL → Answers.** The open-source AI agent for MongoDB.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-mango--ai-orange.svg)](https://pypi.org/project/mango-ai)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescobellingeri/mango-ai/blob/main/notebooks/mango_quickstart.ipynb#scrollTo=v-JUu86PiC_r)

> Ask your MongoDB database anything in plain language. Mango translates your question into an MQL query, runs it, and gives you a clear answer — with memory that improves over time.

---

## What You Get

Ask a question in natural language and get back:

**1. Real-time tool call visibility** — see exactly which collections and queries Mango uses

**2. A natural language answer** — summarised from the actual query results

**3. A system that learns** — every successful query is stored in memory and reused for similar questions in the future

All streamed in real-time over SSE to your frontend.

---

## Why Mango?

Vanna.ai has solved text-to-SQL elegantly. MongoDB has no equivalent. The challenges are fundamentally different:

- **No explicit schema** — collections have no DDL. Mango infers schema by sampling documents.
- **Nested documents** — queries must navigate arrays, subdocuments, and dotted paths.
- **Aggregation pipelines** — complex analytics require multi-stage JSON pipelines, not flat SQL strings.
- **No JOINs** — relationships are handled via `$lookup` or application-level references.

Mango is the first production-grade framework for natural language interaction with MongoDB.

---

## Get Started

### Install

```bash
pip install mango[anthropic]   # Claude
pip install mango[openai]      # GPT
pip install mango[gemini]      # Gemini
pip install mango[all]         # all providers + ChromaDB memory
```

### Minimal setup (3 minutes)

```python
from mango import MangoAgent
from mango.tools import build_mongo_tools
from mango.tools.base import ToolRegistry
from mango.servers.fastapi import MangoFastAPIServer
from mango.integrations.anthropic import AnthropicLlmService
from mango.integrations.mongodb import MongoRunner
from mango.integrations.chromadb import ChromaAgentMemory

# 1. Connect your database
db = MongoRunner()
db.connect("mongodb://localhost:27017/mydb")

# 2. Choose your LLM
llm = AnthropicLlmService(model="claude-sonnet-4-6", api_key="...")

# 3. Add memory (optional but recommended)
memory = ChromaAgentMemory(persist_dir="./mango_memory")

# 4. Register tools and create agent
tools = ToolRegistry()
for tool in build_mongo_tools(db, memory):
    tools.register(tool)

agent = MangoAgent(llm_service=llm, tool_registry=tools, db=db, agent_memory=memory)

# 5. Run the server
MangoFastAPIServer(agent).run()  # http://localhost:8000
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
│  1. Search memory for similar past queries   │
│  2. Build system prompt with schema context  │
│  3. LLM decides which tools to call          │
│  4. Execute tools against MongoDB            │
│  5. Feed results back to LLM                 │
│  6. Stream natural language answer           │
│  7. Auto-save successful queries to memory   │
└─────────────────────────────────────────────┘
      │
      ▼
SSE stream → your frontend
```

**The learning loop:** steps 1 and 7 make Mango smarter over time. A novel question triggers deep LLM reasoning. A similar question already in memory gets answered faster and more accurately.

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

Mango learns from every successful interaction using ChromaDB as a vector store.

**How it works:**

- After each successful tool call, Mango automatically saves the `(question, tool, args, result)` tuple
- On the next similar question, the saved example is injected as a few-shot prompt
- The more you use it, the faster and more accurate it gets

**Pre-load your domain knowledge:**

```python
# Teach Mango about your business terminology
memory.save_text("'active customer' means a customer who placed an order in the last 90 days")
memory.save_text("'revenue' always refers to the total_amount field in the orders collection")
memory.save_text("the 'status' field uses: 1=pending, 2=shipped, 3=delivered, 4=cancelled")
```

---

## Available Tools

| Tool | Description |
|------|-------------|
| `list_collections` | List all collections. Grouped view for large databases (100+ collections). |
| `search_collections` | Search collections by name pattern (supports glob: `order*`, `*_log`). |
| `describe_collection` | Full schema for a collection: field types, frequencies, indexes, references. |
| `collection_stats` | Document count and storage size for a collection. |
| `run_mql` | Execute a read-only MongoDB query: `find`, `aggregate`, `count`, `distinct`. |
| `search_saved_correct_tool_uses` | Search memory for similar past interactions. |
| `save_text_memory` | Save free-form knowledge about the database for future queries. |

> **Read-only by design.** `run_mql` only accepts `find`, `aggregate`, `count`, `distinct`. Write operations are rejected at the tool level.

---

## Pluggable Architecture

Mango is built on abstract interfaces — swap any component without touching your agent code.

### LLM Providers

```python
from mango.integrations.anthropic import AnthropicLlmService
from mango.integrations.openai import OpenAILlmService
from mango.integrations.google import GeminiLlmService

llm = AnthropicLlmService(model="claude-sonnet-4-6")
llm = OpenAILlmService(model="gpt-5.4")
llm = GeminiLlmService(model="gemini-3.1-pro-preview")
```

### Custom Tools

Extend Mango with your own tools:

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

The `MemoryService` ABC makes it easy to plug in any vector store:

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
| LLM (secondary) | openai (GPT-4), google-generativeai (Gemini) |
| Vector store | chromadb 1.x |
| Server | FastAPI + uvicorn |
| CLI | rich + prompt_toolkit |
| Data | pandas |
| Testing | pytest + mongomock |

---

## Roadmap

- [x] MongoDB backend with schema introspection
- [x] Pluggable LLM providers (Anthropic, OpenAI, Gemini)
- [x] ChromaDB agent memory with auto-save
- [x] Streaming FastAPI server (SSE)
- [x] Adaptive collection grouping for large databases
- [x] Multi-turn conversation with automatic pruning
- [ ] `ValidatorTool` — pre-execution MQL validation
- [ ] `ExplainQueryTool` — pipeline explanation in plain language
- [ ] `VisualizeDataTool` — charts and tables in CLI
- [ ] Memory export/import (JSON)
- [ ] Atlas Vector Search backend
- [ ] Redis backend (experimental)

---

## Contributing

Contributions are welcome. Please open an issue before submitting a large PR so we can discuss the approach.

```bash
git clone https://github.com/francesco-bellingeri/mango
cd mango
pip install -e ".[dev]"
pytest
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with ❤️ by Francesco Bellingeri** | Inspired by [Vanna.ai](https://vanna.ai)
