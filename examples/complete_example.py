import asyncio
import os

from mango import MangoAgent
from mango.tools import (
    ToolRegistry,
    ListCollectionsTool,
    SearchCollectionsTool,
    DescribeCollectionTool,
    CollectionStatsTool,
    RunMQLTool,
    SaveTextMemoryTool,
    DeleteLastMemoryEntryTool,
)
from mango.servers.fastapi import MangoFastAPIServer
from mango.integrations import GeminiLlmService, AnthropicLlmService, OpenAILlmService, OllamaLlmService, MongoRunner, ChromaAgentMemory
from dotenv import load_dotenv

load_dotenv()

# Configure your LLM
llm = AnthropicLlmService(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Configure your database
db = MongoRunner()
db.connect(os.getenv("MONGODB_URI", "mongodb://localhost:27017/mango_marketplace"))

# Configure your agent memory
agent_memory = ChromaAgentMemory(
    persist_dir=".mango_memory",
)

# Register tools
tools = ToolRegistry()
tools.register(ListCollectionsTool(db))
tools.register(SearchCollectionsTool(db))
tools.register(DescribeCollectionTool(db))
tools.register(CollectionStatsTool(db))
tools.register(RunMQLTool(db))
tools.register(SaveTextMemoryTool(agent_memory))

# Create your agent
agent = MangoAgent(
    llm_service=llm,
    tool_registry=tools,
    db=db,
    agent_memory=agent_memory,
    introspect=False,
)

# Wire up the delete tool AFTER agent creation so it can reference agent state
tools.register(DeleteLastMemoryEntryTool(agent_memory, lambda: agent._last_memory_entry_id))

# Pre-load the agent memory from the trainingset JSONL (schema notes + verified
# tool-use examples). Guarded so restarts don't duplicate entries.
# TRAINING_FILE = os.getenv("TRAINING_FILE", "examples/trainingset_marketplace.jsonl")
# if agent_memory.training_count() == 0:
#     from mango.servers.cli.main import _load_training_file

#     print(f"Pre-loading training data from {TRAINING_FILE}…")
#     asyncio.run(_load_training_file(agent_memory, TRAINING_FILE))
# else:
#     print(f"Memory already has {agent_memory.training_count()} training entries — skipping load.")

# Run the server
server = MangoFastAPIServer(agent)
server.run()  # http://localhost:8000
