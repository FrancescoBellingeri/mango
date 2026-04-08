# Quickstart

Get Mango running in 5 minutes.

!!! tip "Try it in the browser"
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescobellingeri/mango-ai/blob/main/notebooks/mango_quickstart.ipynb)

    No local setup needed — open the notebook and run it directly in Google Colab.

---

## 1. Install

Choose your LLM provider:

=== "Anthropic (Claude)"
    ```bash
    pip install mango-ai[anthropic]
    ```

=== "OpenAI (GPT)"
    ```bash
    pip install mango-ai[openai]
    ```

=== "Google (Gemini)"
    ```bash
    pip install mango-ai[gemini]
    ```

=== "All providers"
    ```bash
    pip install mango-ai[all]
    ```

---

## 2. Set environment variables

```bash
export MONGODB_URI="mongodb://localhost:27017/mydb"
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY / GEMINI_API_KEY
```

Or use a `.env` file — Mango loads it automatically via `python-dotenv`.

---

## 3. Run the server

=== "Anthropic (Claude)"
    ```python
    import os
    from mango import MangoAgent
    from mango.integrations.anthropic import AnthropicLlmService
    from mango.integrations.mongodb import MongoRunner
    from mango.integrations.chromadb import ChromaAgentMemory
    from mango.tools import ToolRegistry, build_mongo_tools
    from mango.servers.fastapi import MangoFastAPIServer

    db = MongoRunner()
    db.connect(os.getenv("MONGODB_URI"))

    llm = AnthropicLlmService(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    memory = ChromaAgentMemory(persist_dir="./mango_memory")

    tools = ToolRegistry()
    for tool in build_mongo_tools(db, memory):
        tools.register(tool)

    agent = MangoAgent(llm_service=llm, tool_registry=tools, db=db, agent_memory=memory)
    MangoFastAPIServer(agent).run()  # http://localhost:8000
    ```

=== "OpenAI (GPT)"
    ```python
    import os
    from mango import MangoAgent
    from mango.integrations.openai import OpenAiLlmService
    from mango.integrations.mongodb import MongoRunner
    from mango.integrations.chromadb import ChromaAgentMemory
    from mango.tools import ToolRegistry, build_mongo_tools
    from mango.servers.fastapi import MangoFastAPIServer

    db = MongoRunner()
    db.connect(os.getenv("MONGODB_URI"))

    llm = OpenAiLlmService(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    memory = ChromaAgentMemory(persist_dir="./mango_memory")

    tools = ToolRegistry()
    for tool in build_mongo_tools(db, memory):
        tools.register(tool)

    agent = MangoAgent(llm_service=llm, tool_registry=tools, db=db, agent_memory=memory)
    MangoFastAPIServer(agent).run()  # http://localhost:8000
    ```

=== "Google (Gemini)"
    ```python
    import os
    from mango import MangoAgent
    from mango.integrations.google import GeminiLlmService
    from mango.integrations.mongodb import MongoRunner
    from mango.integrations.chromadb import ChromaAgentMemory
    from mango.tools import ToolRegistry, build_mongo_tools
    from mango.servers.fastapi import MangoFastAPIServer

    db = MongoRunner()
    db.connect(os.getenv("MONGODB_URI"))

    llm = GeminiLlmService(
        model="gemini-3.1-pro-preview",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    memory = ChromaAgentMemory(persist_dir="./mango_memory")

    tools = ToolRegistry()
    for tool in build_mongo_tools(db, memory):
        tools.register(tool)

    agent = MangoAgent(llm_service=llm, tool_registry=tools, db=db, agent_memory=memory)
    MangoFastAPIServer(agent).run()  # http://localhost:8000
    ```

=== "All providers"
    ```python
    import os
    from mango import MangoAgent
    from mango.llm.factory import build_llm
    from mango.integrations.mongodb import MongoRunner
    from mango.integrations.chromadb import ChromaAgentMemory
    from mango.tools import ToolRegistry, build_mongo_tools
    from mango.servers.fastapi import MangoFastAPIServer

    db = MongoRunner()
    db.connect(os.getenv("MONGODB_URI"))

    # Switch provider via environment variable: anthropic | openai | gemini
    llm = build_llm(provider=os.getenv("MANGO_PROVIDER", "openai"))

    memory = ChromaAgentMemory(persist_dir="./mango_memory")

    tools = ToolRegistry()
    for tool in build_mongo_tools(db, memory):
        tools.register(tool)

    agent = MangoAgent(llm_service=llm, tool_registry=tools, db=db, agent_memory=memory)
    MangoFastAPIServer(agent).run()  # http://localhost:8000
    ```

---

## 4. Ask your first question

```bash
curl -X POST http://localhost:8000/api/v1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How many documents are in the largest collection?"}' \
  --no-buffer
```

You'll see a stream of SSE events in real-time:

```
data: {"type": "tool_call",   "tool_name": "list_collections", ...}
data: {"type": "tool_result", "tool_name": "list_collections", ...}
data: {"type": "tool_call",   "tool_name": "collection_stats", ...}
data: {"type": "answer",      "text": "The largest collection is 'orders' with 1,247,832 documents."}
data: {"type": "done",        "iterations": 2, "input_tokens": 1820, "output_tokens": 94}
```

---

## 5. Ask in Python (without the server)

You can also call the agent directly without FastAPI:

```python
import asyncio

response = asyncio.run(agent.ask("How many orders were placed last week?"))
print(response.answer)
```

---

## Demo database

Don't have a MongoDB database handy? Use the free Atlas sample dataset.

Run this script to create a free cluster with sample data (films, restaurants, Airbnb listings):

```bash
./scripts/setup_atlas_demo.sh
```

It will print a ready-to-use `MONGODB_URI` at the end.
