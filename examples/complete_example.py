import os

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
from mango.integrations import GeminiLlmService, AnthropicLlmService, OpenAiLlmService, OllamaLlmService, MongoRunner, ChromaAgentMemory
from dotenv import load_dotenv

load_dotenv()

# Configure your LLM
llm = OpenAiLlmService(
    model="gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Configure your database
db = MongoRunner()
db.connect(os.getenv("MONGODB_URI", "mongodb://localhost:27017/mydb"))

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
tools.register(SearchSavedCorrectToolUsesTool(agent_memory))
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

# Run the server
server = MangoFastAPIServer(agent)
server.run()  # http://localhost:8000
