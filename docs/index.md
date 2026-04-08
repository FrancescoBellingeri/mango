# 🥭 Mango — MongoDB AI Agent

**Natural language → MQL → Answers.** The open-source AI agent for MongoDB.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/francescobellingeri/mango-ai/blob/main/LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-mango--ai-orange.svg)](https://pypi.org/project/mango-ai)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescobellingeri/mango-ai/blob/main/notebooks/mango_quickstart.ipynb)

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

---

## Technology Stack

| Component | Library |
|-----------|---------|
| MongoDB driver | pymongo 4.x |
| Vector store | chromadb 1.x |
| Server | FastAPI + uvicorn |
| Data | pandas |
| Testing | pytest |

**LLM Providers:** Anthropic Claude, OpenAI GPT, Google Gemini — all pluggable.

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
- [ ] Memory export/import (JSON)
- [ ] Atlas Vector Search backend
