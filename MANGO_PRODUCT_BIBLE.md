# 🥭 MANGO — Product Bible

**Mongo Agent for Natural-language Operations**

> A Vanna-like AI agent framework for NoSQL databases.
> MongoDB-first, pluggable architecture, read-only by design.

**v1.0 — April 2026**

---

## Table of Contents

1. [Vision & Overview](#1-vision--overview)
2. [Architecture](#2-architecture)
3. [Abstract Interfaces (Contracts)](#3-abstract-interfaces-contracts)
4. [Tools in Detail](#4-tools-in-detail)
5. [Schema Context Engine](#5-schema-context-engine)
6. [MQL Generation Strategy](#6-mql-generation-strategy)
7. [Prompt Engineering](#7-prompt-engineering)
8. [Agent Memory](#8-agent-memory)
9. [CLI Interface](#9-cli-interface)
10. [Pluggability Guide](#10-pluggability-guide)
11. [Technology Stack](#11-technology-stack)
12. [Development Roadmap](#12-development-roadmap)
13. [Conventions for Claude Code](#13-conventions-for-claude-code)
14. [Features to Implement](#14-features-to-implement)

---

## 1. Vision & Overview

### 1.1 What is Mango?

Mango (Mongo Agent for Natural-language Operations) is an open-source AI agent framework that enables users to interact with NoSQL databases using natural language. Inspired by Vanna.ai 2.0, Mango replicates the same proven architecture — tool registry, agent memory, pluggable LLM, schema context — but adapts it to the fundamentally different world of document-oriented and NoSQL databases.

### 1.2 Why Mango Exists

Vanna.ai has solved the text-to-SQL problem elegantly. However, MongoDB and other NoSQL databases have no equivalent solution. The challenges are different and harder:

- **No explicit schema:** MongoDB collections have no DDL. The schema must be inferred by sampling documents.
- **Structured queries:** MQL queries are nested JSON/dict structures, not flat text strings. Aggregation pipelines are lists of stages.
- **Polymorphic documents:** Documents in the same collection can have different shapes, optional fields, and nested subdocuments.
- **No JOINs:** Relationships are handled via `$lookup` in aggregation or application-level references, not SQL JOINs.

Existing tools (LangChain chains, custom GPT wrappers) are fragile scripts, not production-grade frameworks. Mango aims to be the definitive solution.

### 1.3 Design Principles

- **Vanna-like architecture:** Tool registry, agent memory, pluggable LLM service, user permissions. If it works for Vanna, we adopt it.
- **Read-only by design:** Mango never modifies data. All operations are read-only (`find`, `aggregate`, `count`, `distinct`). This is a deliberate safety constraint for production use.
- **MongoDB-first, pluggable:** We build and perfect MongoDB support first, but the architecture uses abstract backend interfaces so Redis, Cassandra, DynamoDB, etc. can be added later.
- **Learn over time:** Agent memory stores successful question-to-MQL pairs. The system gets better with use, exactly like Vanna.
- **Schema-aware:** Automatic schema introspection gives the LLM the context it needs to generate correct queries.
- **CLI-first:** Terminal interaction. No web UI for now. The agent is a Python library + CLI tool.

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Interface                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                     MANGO AGENT                          │
│                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │  LLM Service │  │ Tool Registry │  │ Schema       │  │
│  │  (Claude,    │  │ (dispatch +   │  │ Context      │  │
│  │   GPT, etc.) │  │  permissions) │  │ (sampled +   │  │
│  │              │  │               │  │  inferred)   │  │
│  └──────────────┘  └───────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                     TOOL LAYER                           │
│                                                          │
│  🔍 RunMqlTool           (find / aggregate / count)      │
│  🗂️ SchemaIntrospector   (infer types + refs)            │
│  📊 VisualizeDataTool    (charts + tables)               │
│  📝 ExplainQueryTool     (pipeline in plain language)    │
│  ✅ ValidatorTool        (pre-execution checks)          │
│  🧠 MemoryTools          (save + search)                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           NoSQL Backend Interface (ABC)                   │
│  connect() / execute_query() / introspect_schema()       │
│  get_sample_docs() / list_collections() / get_indexes()  │
└──────┬─────────────────┬─────────────────┬──────────────┘
       │                 │                 │
       ▼                 ▼                 ▼
 ┌────────────┐   ┌────────────┐   ┌────────────────┐
 │ Mongo      │   │ Redis      │   │ Cassandra      │
 │ Backend    │   │ Backend    │   │ Backend         │
 │ (pymongo)  │   │ (future)   │   │ (future)        │
 └────────────┘   └────────────┘   └────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Vector Store       │
              │  (ChromaDB)         │
              │  question → MQL     │
              │  pairs + schema     │
              │  embeddings         │
              └─────────────────────┘
```

### 2.2 Request Flow

Every user question follows this exact pipeline:

**Step 1 — Memory Search:** The agent searches the vector store for previously successful question-MQL pairs similar to the current question.

**Step 2 — Prompt Composition:** The system prompt is assembled with: schema context (inferred types, sample documents, indexes), few-shot examples from memory, the user's natural language question.

**Step 3 — LLM Generation:** The LLM receives the composed prompt and decides which tool(s) to call. Typically it calls `RunMqlTool` with a structured JSON payload.

**Step 4 — Validation:** Before execution, the `ValidatorTool` checks: does the collection exist? Do the referenced fields exist in the inferred schema? Is the JSON structure syntactically valid? Are all MQL operators valid?

**Step 5 — Execution:** `RunMqlTool` executes the query against the database and returns results as a pandas DataFrame.

**Step 6 — Response:** The LLM formats a natural-language answer. Optionally calls `VisualizeDataTool` for charts or `ExplainQueryTool` for pipeline explanations.

**Step 7 — Memory Save:** If the user confirms the result is correct, the question-MQL pair is saved to agent memory for future use.

> **Key Insight:** Steps 1 and 7 form the learning loop. The more the system is used, the better it gets. A novel question triggers deep LLM reasoning; a similar question found in memory triggers fast adaptation from stored examples. This is the same pattern that makes Vanna effective.

### 2.3 Project Structure

```
mango/                              # repository root
├── app.py                          # entrypoint — wires services & starts server
├── pyproject.toml                  # PEP 621 metadata & dependencies
├── requirements.txt
├── README.md
├── MANGO_PRODUCT_BIBLE.md
├── .env                            # local secrets (not committed)
│
├── frontend/                       # Nuxt 3 UI
│   ├── app/
│   │   ├── app.vue
│   │   ├── assets/css/main.css
│   │   └── pages/index.vue
│   ├── public/
│   ├── nuxt.config.ts
│   └── package.json
│
├── mango/                          # core Python package
│   ├── __init__.py                 # public API (MangoAgent, etc.)
│   │
│   ├── agent/                      # orchestrator
│   │   ├── agent.py                # MangoAgent — main entry point
│   │   └── prompt_builder.py       # system prompt assembly
│   │
│   ├── core/                       # shared types & tool registry
│   │   ├── types.py                # QueryRequest, ToolResult, SchemaInfo, …
│   │   └── registry.py             # ToolRegistry — register & dispatch tools
│   │
│   ├── llm/                        # LLM abstraction
│   │   ├── base.py                 # LLMService (ABC)
│   │   └── models.py               # shared message/response models
│   │
│   ├── memory/                     # agent memory abstraction
│   │   ├── base.py                 # AgentMemory (ABC)
│   │   └── models.py               # MemoryEntry dataclass
│   │
│   ├── nosql_runner/               # database backend abstraction
│   │   └── base.py                 # NoSQLRunner (ABC)
│   │
│   ├── tools/                      # built-in tools
│   │   ├── base.py                 # BaseTool (ABC)
│   │   └── mongo_tools.py          # ListCollectionsTool, DescribeCollectionTool, RunMQLTool
│   │
│   ├── integrations/               # concrete implementations
│   │   ├── mongodb.py              # MongoRunner (NoSQLRunner)
│   │   ├── chromadb.py             # ChromaAgentMemory (AgentMemory)
│   │   ├── google.py               # GeminiLlmService (LLMService)
│   │   ├── openai.py               # OpenAILlmService (LLMService)
│   │   └── anthropic.py            # AnthropicLlmService (LLMService)
│   │
│   └── servers/                    # server adapters
│       ├── cli/
│       │   └── main.py             # MangoCliServer — interactive terminal
│       └── fastapi/
│           ├── main.py             # MangoFastAPIServer — uvicorn wrapper
│           ├── routes.py           # /health, /api/v1/ask/stream (SSE)
│           └── models.py           # AskRequest, HealthResponse (Pydantic)
│
└── tests/
    ├── test_agent.py
    ├── test_memory.py
    ├── test_mongo_tools.py
    ├── test_prompt_builder.py
    ├── test_tool_registry.py
    ├── test_types.py
    ├── test_coerce_dates.py
    ├── test_context_enhancer.py
    ├── test_memory_sanity.py
    ├── test_mongo_backend.py
    └── integration/
        └── test_workflow.py
```

---

## 3. Abstract Interfaces (Contracts)

These are the ABC classes that enable pluggability. Every concrete implementation must respect these contracts.

### 3.1 NoSQLBackend (`backends/base.py`)

The abstract interface for all database backends:

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `connect(connection_string: str, **kwargs) -> None` | Establish database connection |
| `execute_query` | `execute_query(operation: QueryRequest) -> pd.DataFrame` | Execute a read-only query, return DataFrame |
| `introspect_schema` | `introspect_schema() -> dict[str, SchemaInfo]` | Return inferred schema for all collections |
| `get_sample_documents` | `get_sample_documents(collection: str, n: int = 5) -> list[dict]` | Return N sample documents |
| `list_collections` | `list_collections() -> list[str]` | List all collection/keyspace names |
| `get_indexes` | `get_indexes(collection: str) -> list[dict]` | Return index definitions |

### 3.2 QueryRequest (`core/types.py`)

The standardized query format that all backends receive. This is the key to pluggability:

```python
@dataclass
class QueryRequest:
    operation: Literal["find", "aggregate", "count", "distinct"]
    collection: str
    filter: dict | None = None        # {field: {$op: value}}
    pipeline: list[dict] | None = None # [{$match: ...}, {$group: ...}]
    projection: dict | None = None     # {field: 1, _id: 0}
    sort: dict | None = None           # {field: 1 or -1}
    limit: int | None = None
    distinct_field: str | None = None  # for operation='distinct'
```

### 3.3 LLMService (`llm/base.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `chat` | `chat(messages: list[dict], tools: list[ToolDef]) -> LLMResponse` | Send messages + tool definitions, get response with possible tool calls |
| `get_model_name` | `get_model_name() -> str` | Return the model identifier |

### 3.4 AgentMemory (`memory/base.py`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `save(question: str, tool_name: str, tool_args: dict, result_summary: str) -> None` | Store a successful interaction |
| `search` | `search(question: str, n: int = 3) -> list[MemoryEntry]` | Semantic search for similar past questions |
| `save_text` | `save_text(text: str, metadata: dict) -> None` | Store arbitrary text (schema docs, glossary) |
| `delete` | `delete(entry_id: str) -> None` | Remove a memory entry |
| `list_all` | `list_all() -> list[MemoryEntry]` | List all stored memories |

---

## 4. Tools in Detail

Each tool is a self-contained capability registered with the `ToolRegistry`. The LLM decides which tool to call based on the user's question.

### 4.1 RunMqlTool

**Purpose:** Execute read-only MongoDB queries and return results as a pandas DataFrame.

**Input from LLM:** A `QueryRequest` JSON specifying operation type, collection, filter/pipeline, projection, sort, limit.

**Supported operations:**

- `find` — Standard document queries with filter, projection, sort, limit
- `aggregate` — Aggregation pipelines with `$match`, `$group`, `$project`, `$sort`, `$limit`, `$unwind`, `$lookup`, `$facet`, etc.
- `count` — Count documents matching a filter
- `distinct` — Get distinct values for a field

**Output:** pandas DataFrame (for tabular display and charting compatibility) or scalar value (for count).

> ⛔ **Safety Constraint:** RunMqlTool MUST reject any operation that is not read-only. It should explicitly **allowlist** operations (`find`, `aggregate`, `count`, `distinct`), NOT blocklist. The tool must validate the operation type before execution.

### 4.2 SchemaIntrospectorTool

**Purpose:** Infer and cache the schema of all collections in the database. This is the most critical tool for MongoDB because there is no explicit DDL.

#### Introspection Strategy

**Phase 1 — Sampling:** For each collection, sample N documents (configurable, default 100). Use a mix of `collection.find().limit(N)` and `collection.aggregate([{$sample: {size: N}}])` to get both sequential and random samples.

**Phase 2 — Schema Merging:** Traverse all sampled documents recursively. For each field path (including nested paths like `address.city`), record: field name and full dotted path, observed types (string, int, float, bool, ObjectId, date, array, subdocument), frequency of presence (0.0 to 1.0), for arrays: element types, for subdocuments: recurse.

**Phase 3 — Index Analysis:** Call `collection.index_information()` to get all indexes. Map indexed fields to the inferred schema.

**Phase 4 — Reference Detection:** Heuristic: if a field name ends with `_id` or `Id` and a collection with the matching name exists (e.g. `user_id` + `users` collection), flag it as a probable reference.

**Phase 5 — Caching:** Store the complete inferred schema in memory. Refresh only on explicit request or on a configurable TTL.

#### SchemaInfo Data Structure

```python
@dataclass
class FieldInfo:
    name: str
    path: str                          # dotted path: 'address.city'
    types: list[str]                   # ['string', 'null']
    frequency: float                   # 0.0 - 1.0
    is_indexed: bool
    is_unique: bool
    is_reference: bool
    reference_collection: str | None
    sub_fields: list[FieldInfo] | None  # for subdocuments
    array_element_types: list[str] | None

@dataclass
class SchemaInfo:
    collection_name: str
    document_count: int
    fields: list[FieldInfo]
    indexes: list[dict]
    sample_documents: list[dict]        # 3-5 representative samples
```

### 4.3 ValidatorTool

**Purpose:** Validate a generated MQL query before execution. This catches LLM hallucinations early.

#### Validation Checks

| Check | What It Does | On Failure |
|-------|-------------|------------|
| Collection exists | Verify the target collection is in `list_collections()` | Error: suggest similar collection names |
| Field exists | Verify referenced fields exist in inferred schema | Warning: field not found in sampled schema (may still exist) |
| JSON validity | Parse the filter/pipeline as valid JSON/dict | Error: show parse error location |
| Operator validity | All `$operators` are valid MQL operators | Error: show invalid operator + suggest correct one |
| Pipeline stage order | Common sense checks (`$group` before `$unwind` that depends on it, etc.) | Warning: suggest reordering |
| Type compatibility | Field type matches the filter value type | Warning: possible type mismatch |

> **Note on Validation Severity:** Checks are categorized as **Error** (blocks execution) or **Warning** (proceeds with caution, notifies LLM). This distinction is important because MongoDB's schemaless nature means a field might exist in some documents but not appear in our sample.

### 4.4 ExplainQueryTool

**Purpose:** Take a generated MQL pipeline and produce (a) a natural-language explanation of what the pipeline does step by step, and (b) optionally call MongoDB's `.explain()` to provide execution plan details (indexes used, documents examined, etc.).

This is useful for transparency and for helping users learn MongoDB query patterns.

### 4.5 VisualizeDataTool

**Purpose:** Generate charts and formatted tables from query results. Takes a DataFrame and a visualization request from the LLM.

**Supported outputs:** bar chart, line chart, pie chart, scatter plot, formatted table. In CLI mode, charts are rendered via matplotlib and saved as images, or displayed inline if the terminal supports it. Tables are rendered via `rich`.

### 4.6 MemoryTools (Save + Search)

Directly modeled on Vanna's agent memory pattern:

- **SearchSavedCorrectToolUsesTool:** Called at the beginning of every request. Embeds the user's question and performs similarity search against stored entries. Returns top-N matches as few-shot examples for the LLM.
- **SaveQuestionToolArgsTool:** Called when a query is confirmed correct. Stores the `(question, tool_name, tool_args, result_summary)` tuple with an embedding of the question.
- **SaveTextMemoryTool:** Stores free-text memories: business glossary definitions, schema documentation, domain-specific knowledge that helps the LLM generate better queries.

---

## 5. Schema Context Engine

This is the component that differentiates Mango from Vanna the most. In SQL, you can feed the LLM a `CREATE TABLE` statement and it knows everything. In MongoDB, we must build this context ourselves.

### 5.1 What the LLM Receives

Before generating any query, the LLM's system prompt includes:

#### Collection Overview

```
Database: ecommerce
Collections:
  - orders (245,891 documents)
  - customers (12,430 documents)
  - products (3,217 documents)
  - reviews (89,012 documents)
```

#### Schema per Collection

```
Collection: orders
Fields:
  _id          ObjectId  (100%)  [indexed, unique]
  customer_id  ObjectId  (100%)  [indexed] -> customers
  status       string    (100%)  values: ['pending','shipped','delivered','cancelled']
  items        array     (100%)  of subdocument:
    product_id  ObjectId  -> products
    quantity    int
    price       float
  total_amount  float    (100%)
  created_at    date     (100%)  [indexed]
  shipping      subdoc   (95%)
    address     string
    city        string
    country     string
    tracking_no string   (72%)
```

#### Sample Documents

2-3 actual documents per collection (sanitized if needed), so the LLM sees the real shape of the data.

#### Indexes

```
Indexes on 'orders':
  _id_: { _id: 1 } (unique)
  customer_idx: { customer_id: 1 }
  date_status_idx: { created_at: -1, status: 1 }
```

### 5.2 Schema Refresh Strategy

- Full introspection runs on first connection and is cached.
- User can trigger manual refresh via CLI command: `/refresh-schema`
- Optional TTL-based refresh (configurable, default: disabled).
- Incremental refresh per-collection is supported for large databases.

### 5.3 Business Glossary

Users can add domain-specific context to improve query generation:

```python
# Via CLI or programmatically:
agent.memory.save_text(
    "'active customer' means a customer who placed an order in the last 90 days",
    metadata={"type": "glossary"}
)
agent.memory.save_text(
    "'revenue' always refers to the total_amount field in the orders collection",
    metadata={"type": "glossary"}
)
```

These text memories are included in the schema context when semantically relevant to the user's question.

---

## 6. MQL Generation Strategy

Generating correct MongoDB queries from natural language is the hardest problem Mango solves. Here's the multi-level strategy.

### 6.1 Level 1 — Rich Schema Context

The LLM receives the full inferred schema with types, frequencies, sample documents, and indexes (see Section 5). This gives it the "vocabulary" of the database.

### 6.2 Level 2 — Few-Shot from Memory

If similar questions exist in agent memory, the LLM receives 2-3 examples of `(question, MQL)` pairs. This is the single biggest accuracy booster. Example:

```python
# Memory match found (similarity: 0.91):
# Question: "How many orders per customer last month?"
# Tool: RunMqlTool
# Args:
{
  "operation": "aggregate",
  "collection": "orders",
  "pipeline": [
    {"$match": {"created_at": {"$gte": "ISODate(2026-03-01)"}}},
    {"$group": {"_id": "$customer_id", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
  ]
}
```

### 6.3 Level 3 — Pre-Execution Validation

`ValidatorTool` catches structural errors before they hit the database (see Section 4.3).

### 6.4 Level 4 — Retry with Error Feedback

If execution fails, the MongoDB error message is fed back to the LLM for a second attempt. Maximum 2 retries to avoid infinite loops.

```python
# Retry flow:
# 1. LLM generates query -> ValidatorTool OK -> Execute -> MongoDB Error
# 2. Error fed to LLM: "FieldPath '$items.price' not found"
# 3. LLM corrects: uses $unwind before accessing nested field
# 4. Re-execute -> Success
```

### 6.5 Level 5 — Date/Time Handling

Dates are notoriously tricky. The system prompt explicitly instructs the LLM:

- Always use `ISODate()` format for date comparisons
- "Last month" = `$gte: ISODate(first day of previous month)`
- The current date/time is injected into the system prompt at every request
- Timezone-aware by default (configurable)

---

## 7. Prompt Engineering

The system prompt is the backbone of Mango.

### 7.1 System Prompt Structure

```
SYSTEM PROMPT:

[1] ROLE
You are a MongoDB query expert agent. You help users query
NoSQL databases by generating correct MQL queries.

[2] RULES
- ONLY generate read-only queries (find, aggregate, count, distinct)
- NEVER generate insert, update, delete, drop, or any write operation
- Always use the tools provided, never output raw MQL in text
- When unsure about schema, use SchemaIntrospectorTool first
- Prefer aggregation pipelines for complex analytics
- Always include a $limit unless the user explicitly wants all results

[3] CURRENT CONTEXT
Current datetime: {datetime.now().isoformat()}
Database: {db_name}

[4] SCHEMA CONTEXT
{formatted_schema}

[5] MEMORY CONTEXT (if available)
Similar questions from past interactions:
{formatted_memory_results}

[6] GLOSSARY (if available)
Domain-specific definitions:
{formatted_glossary}
```

### 7.2 Tool Definitions for the LLM

Each tool is described to the LLM in the standard function-calling format:

```json
{
  "name": "run_mql",
  "description": "Execute a read-only MongoDB query...",
  "parameters": {
    "operation": {"type": "string", "enum": ["find","aggregate","count","distinct"]},
    "collection": {"type": "string"},
    "filter": {"type": "object", "description": "MongoDB filter document"},
    "pipeline": {"type": "array", "description": "Aggregation pipeline stages"},
    "projection": {"type": "object"},
    "sort": {"type": "object"},
    "limit": {"type": "integer", "default": 20}
  }
}
```

---

## 8. Agent Memory

### 8.1 How Memory Works

Agent memory is the learning engine of Mango, directly inspired by Vanna 2.0. It uses a vector store to enable semantic search over past interactions.

#### Storage Format

```python
@dataclass
class MemoryEntry:
    id: str                    # unique identifier
    question: str              # original natural language question
    tool_name: str             # e.g. 'run_mql'
    tool_args: dict            # the exact QueryRequest that worked
    result_summary: str        # brief summary of results
    embedding: list[float]     # vector embedding of the question
    created_at: datetime
    confirmed_by: str | None   # user who confirmed correctness
```

### 8.2 ChromaDB Implementation

- Default collection name: `mango_memory`
- Embedding model: configurable (default: Chroma's default `all-MiniLM-L6-v2`)
- Metadata stored: `tool_name`, `tool_args` (serialized JSON), `collection_name`, `operation_type`
- Similarity search uses cosine distance
- Persistence: ChromaDB persistent client with configurable path

### 8.3 Memory Lifecycle

**Bootstrap:** Users can pre-load the memory with common question-MQL pairs for their domain using the CLI `/train` command or programmatically via `agent.memory.save()`.

**Organic growth:** During normal use, when a query returns correct results, the interaction is automatically added to memory.

**Correction:** If a stored memory is wrong, users can delete it: `/forget <entry_id>`.

**Export/Import:** Memory can be exported as JSON for backup or transfer between environments.

### 8.4 Future: Atlas Vector Search

For production deployments on MongoDB Atlas, agent memory can be stored directly in Atlas using Atlas Vector Search. This keeps everything in the same infrastructure and supports distributed deployments.

---

## 9. CLI Interface

### 9.1 User Experience

```
$ mango connect mongodb://localhost:27017/ecommerce
Connected to ecommerce (5 collections, 350K+ documents)
Schema introspection complete.

mango> How many orders were placed last week?

[RunMqlTool] Executing aggregate on 'orders'...
Pipeline: [{$match: {created_at: {$gte: ISODate('2026-03-25')}}},
           {$count: 'total'}]

Result: 1,247 orders were placed in the last 7 days.

mango> Show me daily revenue trend for March

[RunMqlTool] Executing aggregate on 'orders'...
[VisualizeDataTool] Generating line chart...

[Chart displayed: Daily revenue March 2026]

mango> /explain
The pipeline works in 4 stages:
  1. $match: filters orders from March 2026
  2. $group: groups by day, sums total_amount
  3. $sort: orders by date ascending
  4. $project: formats the output fields
```

### 9.2 CLI Commands

| Command | Description |
|---------|-------------|
| `/connect <uri>` | Connect to a MongoDB instance |
| `/schema` | Display inferred schema for all collections |
| `/schema <collection>` | Display schema for a specific collection |
| `/refresh-schema` | Re-run schema introspection |
| `/forget <id>` | Delete a memory entry |
| `/memories` | List all stored memories |
| `/train` | Enter training mode to pre-load question-MQL pairs |
| `/explain` | Explain the last generated pipeline |
| `/export-memory <path>` | Export memory to JSON file |
| `/import-memory <path>` | Import memory from JSON file |
| `/glossary <text>` | Add a business glossary entry |
| `/help` | Show available commands |
| `/quit` | Exit Mango |

---

## 10. Pluggability Guide

Mango is designed to be extended on three axes: backends, LLM providers, and memory stores.

### 10.1 Adding a New Backend

To add support for a new NoSQL database (e.g., Redis, Cassandra, DynamoDB):

1. Create a new file in `backends/`, e.g. `redis.py`
2. Implement the `NoSQLBackend` ABC (all 6 methods)
3. The hardest part is `execute_query`: translating the standardized `QueryRequest` into the database's native query language
4. For databases without aggregation (e.g. Redis), `execute_query` may need to run multiple commands and aggregate in Python
5. Register the backend in the agent's config

### 10.2 Adding a New LLM Provider

1. Create a new file in `llm/`, e.g. `gemini.py`
2. Implement the `LLMService` ABC (`chat` method + model name)
3. Ensure the provider supports function/tool calling (required for tool dispatch)
4. Map tool call format to/from the provider's specific API format

### 10.3 Adding a New Memory Store

1. Create a new file in `memory/`, e.g. `pinecone.py`
2. Implement the `AgentMemory` ABC (`save`, `search`, `save_text`, `delete`, `list_all`)
3. Handle embedding generation (either use the store's built-in embeddings or bring your own)

---

## 11. Technology Stack

| Component | Library/Tool | Version | Notes |
|-----------|-------------|---------|-------|
| Language | Python | 3.11+ | Type hints, dataclasses, async support |
| MongoDB Driver | pymongo | 4.x | Official MongoDB Python driver |
| LLM (primary) | anthropic | latest | Claude API with tool use |
| LLM (secondary) | openai | latest | GPT-4 with function calling |
| Vector Store | chromadb | 0.5+ | Persistent mode, cosine similarity |
| Data Processing | pandas | 2.x | DataFrame for query results |
| CLI Framework | prompt_toolkit | 3.x | Interactive terminal with history |
| CLI Formatting | rich | 13+ | Tables, syntax highlighting, panels |
| Visualization | matplotlib | 3.x | Charts in CLI |
| Testing | pytest | 8+ | Unit + integration tests |
| Build System | pyproject.toml | — | PEP 621 metadata, optional deps |
| Linting | ruff | latest | Fast Python linter |
| Type Checking | mypy | latest | Static type analysis |

### 11.1 Optional Dependencies

```bash
# Core
pip install mango-ai[mongodb]     # pymongo + core

# LLM providers
pip install mango-ai[anthropic]   # + anthropic SDK
pip install mango-ai[openai]      # + openai SDK

# Full install
pip install mango-ai[all]         # everything
```

---

## 12. Development Roadmap

### Phase 1 — Foundation (MVP)

*Target: working end-to-end flow with MongoDB*

- NoSQLBackend ABC + MongoBackend implementation
- SchemaIntrospectorTool with full sampling and type inference
- LLMService ABC + Anthropic Claude implementation
- RunMqlTool (find + aggregate)
- Basic ValidatorTool (collection/field existence, JSON validity)
- System prompt template with schema context injection
- Basic CLI with prompt_toolkit
- Unit tests for all components

### Phase 2 — Memory & Learning

*Target: system gets better over time*

- AgentMemory ABC + ChromaDB implementation
- SearchSavedCorrectToolUsesTool integration in request flow
- SaveQuestionToolArgsTool with user confirmation flow
- SaveTextMemoryTool for business glossary
- Memory export/import (JSON)
- `/train` command for bulk memory pre-loading

### Phase 3 — Polish & DX

*Target: production-quality developer experience*

- VisualizeDataTool (matplotlib charts in CLI)
- ExplainQueryTool (pipeline explanation + `.explain()`)
- Retry-with-error flow (max 2 retries)
- Advanced validation (type compatibility, pipeline stage order)
- Rich CLI output (tables, syntax-highlighted pipelines, panels)
- Comprehensive error messages and user guidance

### Phase 4 — Expansion

*Target: beyond MongoDB*

- OpenAI LLM provider
- Atlas Vector Search memory backend
- Redis backend (experimental)
- Cassandra backend (experimental)
- DynamoDB backend (experimental)
- Web API (FastAPI) for programmatic access
- Memory analytics (most common queries, accuracy tracking)

---

### Weekly Delivery Schedule

Granular week-by-week plan for social updates and incremental releases.

**Blocco 1 — Reliability (settimane 1–3)**

| Settimana | Feature | LinkedIn angle |
|-----------|---------|----------------|
| 1 ✓ | **ValidatorTool** — validazione strutturale pre-esecuzione | "Mango now catches bad queries before they hit your database" |
| 2 ✓ | **Error Recovery** — retry-with-error, max 2 tentativi, LLM si auto-corregge | "Mango self-corrects when it makes a mistake" |
| 3 | **ExplainQueryTool** — spiegazione step-by-step della pipeline in linguaggio naturale | "Understand exactly what Mango is doing under the hood" |

**Blocco 2 — Memory quality (settimane 4–5)**

| Settimana | Feature | LinkedIn angle |
|-----------|---------|----------------|
| 4 | **Memory export/import + `/train`** — pre-carica knowledge in bulk | "Train Mango on your domain before it even sees a question" |
| 5 | **Ground Truth Validation** — blocca save se risultato diverge dal ground truth | "Mango validates its own memory before it learns" |

**Blocco 3 — Developer experience (settimane 6–8)**

| Settimana | Feature | LinkedIn angle |
|-----------|---------|----------------|
| 6 | **Custom System Prompt** — suffix statico + context per-turn | "Customize Mango for your exact deployment" |
| 7 | **Result Export** — CSV/JSON/Excel da CLI e API | "Take your MongoDB data anywhere" |
| 8 | **VisualizeDataTool** — bar/line/pie/scatter da CLI e stream SSE | "Mango can now chart your data" |

**Blocco 4 — Production readiness (settimane 9–13)**

| Settimana | Feature | LinkedIn angle |
|-----------|---------|----------------|
| 9 | **Collection Access Control** — whitelist/blacklist per agent | "Production-grade security for multi-tenant deployments" |
| 10 | **Custom Middleware** — cache TTL, rate limiting sull'ToolRegistry | "Add caching and rate limiting without touching tool code" |
| 11 | **Conversation Persistence** — sessioni sopravvivono ai restart | "Long-running analyses that survive server restarts" |
| 12 | **Observability** — eventi strutturati, OTel-ready | "Monitor Mango in production with OpenTelemetry" |
| 13 | **Audit Logging** — JSONL append-only, tamper-evident | "Mango for finance and healthcare — immutable audit trail" |

**Blocco 5 — Expansion (settimane 14+)**

| Settimana | Feature | LinkedIn angle |
|-----------|---------|----------------|
| 14 | **Multi-Database Support** — un agent, N database | "Query your entire data landscape in one conversation" |
| 15+ | **Atlas Vector Search**, Redis, Cassandra, DynamoDB | "Mango beyond MongoDB" |

---

## 13. Conventions for Claude Code

> This section is specifically for Claude Code (or any AI coding assistant) working on Mango. Follow these conventions strictly.

### 13.1 Code Style

- Python 3.11+ with full type hints on all function signatures
- Dataclasses for all data structures (`QueryRequest`, `SchemaInfo`, `FieldInfo`, `MemoryEntry`, `ToolResult`)
- ABC classes for all pluggable interfaces (`NoSQLBackend`, `LLMService`, `AgentMemory`)
- No global state — all state lives in the `MangoAgent` instance
- Errors: custom exception hierarchy rooted in `MangoError`
- Logging via standard library `logging` module, not `print()`
- Async-ready: core methods should be async where I/O is involved
- Docstrings: Google style. Every public method gets a docstring

### 13.2 Testing Requirements

- Every ABC method must have a test with a mock implementation
- MongoBackend tests use `mongomock` or a test container
- LLM tests mock the API responses (never call real APIs in tests)
- Memory tests use ChromaDB ephemeral client
- Integration test: full flow from question to result with mocked LLM + real Mongo

### 13.3 Key Architectural Rules

> ⛔ **NEVER BREAK THESE RULES:**
>
> 1. **NEVER** generate or execute write operations (insert, update, delete, drop). The `RunMqlTool` must have an explicit **allowlist** of operations, not a blocklist.
>
> 2. The LLM must **ALWAYS** use tools. It should never output raw MQL in text responses. If it does, the agent must intercept and re-prompt.
>
> 3. All database interaction goes through the `NoSQLBackend` interface. No tool should import `pymongo` directly.
>
> 4. Schema context is **ALWAYS** injected into the system prompt. Never let the LLM guess the schema.
>
> 5. The `QueryRequest` dataclass is the single source of truth for query structure. No tool should accept raw dicts.

### 13.4 File-by-File Build Order

When implementing, follow this order to ensure dependencies are available:

| Order | File | Depends On |
|-------|------|-----------|
| 1 | `core/types.py` | Nothing (pure dataclasses) |
| 2 | `backends/base.py` | `core/types.py` |
| 3 | `backends/mongodb.py` | `backends/base.py` |
| 4 | `llm/base.py` | `core/types.py` |
| 5 | `llm/anthropic.py` | `llm/base.py` |
| 6 | `memory/base.py` | `core/types.py` |
| 7 | `memory/chroma.py` | `memory/base.py` |
| 8 | `tools/validator.py` | `core/types.py`, `backends/base.py` |
| 9 | `tools/run_mql.py` | `core/types.py`, `backends/base.py`, `tools/validator.py` |
| 10 | `tools/schema_introspector.py` | `core/types.py`, `backends/base.py` |
| 11 | `tools/explain_query.py` | `core/types.py` |
| 12 | `tools/visualize.py` | `core/types.py` |
| 13 | `core/registry.py` | `core/types.py`, `tools/*` |
| 14 | `prompts/system.py` | `core/types.py` |
| 15 | `prompts/examples.py` | Nothing |
| 16 | `core/agent.py` | Everything above |
| 17 | `cli/app.py` | `core/agent.py` |

---

## 14. Features to Implement

This section tracks features approved for implementation but not yet built. Each entry describes what the feature is, why it matters for Mango, and the key design decisions to make before implementing.

---

### 14.1 Custom System Prompt

**What:** Allow callers to inject additional instructions into the system prompt at agent creation time or per-turn.

**Why:** Different deployments have different needs — a fintech app might need "always format currency in EUR", a SaaS might need tenant-scoped instructions. Today the system prompt is fully controlled by `build_system_prompt()`.

**Design decisions:**
- `MangoAgent(system_prompt_suffix="...")` for static additions
- Per-turn injection via `ask(question, context="...")` for dynamic additions
- Must not allow overriding the core read-only safety instructions

---

### 14.2 Custom Charts

**What:** A `VisualizeDataTool` that the LLM can call to render query results as charts (bar, line, pie, scatter).

**Why:** Tabular data is hard to read for trend and distribution questions. "Show me daily revenue for March" deserves a chart, not a DataFrame dump.

**Design decisions:**
- Tool receives a DataFrame + chart type + axis config
- CLI: render via `matplotlib` inline or save to file
- API: return base64-encoded image or Vega-Lite spec in the SSE stream
- The LLM decides when to call it — not forced on every query

---

### 14.3 Custom Middleware

**What:** A middleware layer on `ToolRegistry` that wraps tool execution with pluggable hooks. Reference use case: query result caching.

**Why:** Production deployments need cross-cutting concerns (caching, rate limiting, logging, circuit breakers) without touching individual tool implementations.

**Design decisions:**
- `ToolRegistry.add_middleware(fn)` — chain of async callables
- Each middleware receives `(tool_name, kwargs, next)` and returns `ToolResult`
- Built-in middlewares to ship: `CacheMiddleware` (in-memory TTL cache), `RateLimitMiddleware`
- Middleware order matters — document it clearly

---

### 14.4 Error Recovery

**What:** When a tool call fails (bad MQL, timeout, validation error), the agent automatically retries with the error message injected back into the conversation, up to a configurable max.

**Why:** Currently a failed tool call surfaces the raw error to the user. LLMs are good at self-correcting when given the error — we should leverage this.

**Design decisions:**
- `max_retries: int = 2` on `MangoAgent`
- Error message format fed back to LLM must include: tool name, args used, exact error
- Distinguish retryable errors (bad query syntax) from non-retryable (connection down)
- Count retries inside `_run_loop` without resetting `iterations`

---

### 14.5 Observability

**What:** Structured event emission for every agent action — LLM calls, tool executions, memory hits, token usage — consumable by OpenTelemetry, Datadog, or custom sinks.

**Why:** Production users need to monitor latency, cost (tokens), and failure rates. Today everything goes to `logging` with no structure.

**Design decisions:**
- `ObservabilityBackend` ABC with a single `emit(event: AgentEvent)` method
- `AgentEvent` dataclass: `type`, `timestamp`, `duration_ms`, `metadata`
- Built-in: `LoggingObservabilityBackend` (default, uses existing logger), `OTelObservabilityBackend`
- `MangoAgent(observability=MyBackend())` — optional, no-op if not set
- Events to emit: `llm_call`, `tool_call`, `tool_result`, `memory_retrieve`, `memory_store`, `answer`

---

### 14.6 Audit Logging

**What:** Immutable, append-only log of every query executed against the database, with user identity, question, generated MQL, result row count, and timestamp.

**Why:** Required for compliance in regulated industries (finance, healthcare). Separate from observability — audit logs must be tamper-evident and long-lived.

**Design decisions:**
- `AuditLogger` ABC: `log(entry: AuditEntry) -> None`
- `AuditEntry`: `id`, `timestamp`, `user_id`, `question`, `tool_name`, `tool_args`, `row_count`, `success`
- Built-in: `FileAuditLogger` (JSONL append-only), `MongoAuditLogger` (writes to a separate audit collection)
- `MangoAgent(audit_logger=MyLogger(), user_id="...")` — `user_id` threaded through the whole turn
- Must log even failed queries

---

### 14.7 Multi-Database Support

**What:** A single `MangoAgent` instance that can query across multiple connected databases, with the LLM deciding which one to target.

**Why:** Real applications often span multiple databases (e.g. `orders_db` and `analytics_db`). Today one agent = one database.

**Design decisions:**
- `MangoAgent(databases={"orders": mongo1, "analytics": mongo2})`
- Tools receive a `db_name` parameter; the LLM picks the right one
- Schema context includes all databases, clearly labelled
- Memory entries are scoped per database

---

### 14.8 Conversation Persistence

**What:** Save and restore conversation history to/from disk or a database, so sessions survive process restarts.

**Why:** Today conversation history lives in memory — a server restart wipes all active sessions. Long-running sessions (multi-hour analyses) should be resumable.

**Design decisions:**
- `ConversationStore` ABC: `save(session_id, messages)`, `load(session_id) -> list[Message]`
- Built-in: `JsonFileConversationStore`, `MongoConversationStore`
- `MangoAgent(conversation_store=MyStore(), session_id="abc123")`
- Auto-save after each turn; load on first `ask()` if `session_id` matches existing

---

### 14.9 Collection Access Control

**What:** Whitelist or blacklist specific collections per agent instance, preventing the LLM from querying sensitive data even if it tries.

**Why:** Multi-tenant deployments must ensure tenant A cannot access tenant B's collections. Security-sensitive collections (`users`, `payments`) may need to be off-limits for certain agent instances.

**Design decisions:**
- `MangoAgent(allowed_collections=["orders", "products"])` or `denied_collections=["users", "payments"]`
- Enforced at `ToolRegistry` level — `run_mql` raises `ValidationError` before hitting the DB
- Schema introspection respects the same rules — denied collections never appear in the prompt
- Log access attempts on denied collections to audit log

---

### 14.10 Result Export

**What:** Allow users to export query results as CSV, JSON, or Excel directly from the CLI or API.

**Why:** Analysts want to take results into their own tools (Excel, BI platforms). Today results are only returned as text summaries.

**Design decisions:**
- CLI: `/export csv`, `/export json` saves last result to file
- API: `POST /api/v1/ask/export` returns file download
- The raw `pd.DataFrame` from `execute_query` is already the ideal source — just serialize it
- Include the original question and generated MQL in the export metadata

---

### 14.11 Ground Truth Validation

**What:** A pre-persist validation step that runs a newly generated query against a small user-maintained set of questions with verified correct answers, blocking memory saves when results diverge from ground truth.

**Why:** `ValidatorTool` catches structural errors (bad operators, missing fields, invalid JSON). It cannot catch semantic errors — a query that aggregates on the wrong field passes every structural check and returns plausible garbage. Without semantic validation, a single plausible-but-wrong query saved to memory poisons future few-shot examples and compounds over time.

**Design decisions:**
- New `GroundTruthStore` ABC with two methods: `add(question: str, expected: pd.DataFrame | scalar) -> str` and `validate(question: str, actual: pd.DataFrame | scalar) -> ValidationResult`
- Built-in: `InMemoryGroundTruthStore` (default, lost on restart), `MongoGroundTruthStore` (persisted in a dedicated collection), `JsonFileGroundTruthStore`
- `ValidationResult` carries: `passed: bool`, `matched: int`, `total: int`, `failures: list[GroundTruthFailure]`
- Comparison strategy is configurable: exact DataFrame match, numeric tolerance (`rtol`, `atol`), schema-only (shape + dtypes, no values) — different use cases need different tolerance
- Validation runs inside `AgentMemory.save()` as a pre-persist hook — if `GroundTruthStore` is not configured, step is skipped transparently
- `MangoAgent(ground_truth=MyStore())` — optional, no-op if omitted
- CLI: `/ground-truth add` enters interactive mode to record a question + its verified result; `/ground-truth list` shows registered cases; `/ground-truth run` manually triggers full validation suite
- Drift handling: if database schema evolves and a ground-truth case becomes stale, `ValidationResult` includes a `stale_hint` flag (result shape changed) to prompt the user to update the case rather than silently failing
- Performance: ground-truth cases are run sequentially before each save; keep the set small (10–30 cases) — document this constraint clearly
- Scope: validation is per-collection; cases are tagged with `collection_name` so a save to `orders` only runs `orders` ground-truth cases, not the full suite

**Sequencing:** Implement after `ValidatorTool` (14 roadmap). Structural validation (ValidatorTool) runs first; ground-truth validation runs second, only if structural validation passes.

---

*Keep this document in the project root. Update as decisions evolve.*
