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
)
from mango.servers.fastapi import MangoFastAPIServer
from mango.integrations import GeminiLlmService, AnthropicLlmService, OpenAiLlmService, OllamaLlmService, MongoRunner, ChromaAgentMemory
from dotenv import load_dotenv

load_dotenv()

# Configure your LLM
llm = OllamaLlmService(
    model="qwen3.5:4b",
    host="http://localhost:11434",
)

# Configure your database
db = MongoRunner()
db.connect(os.getenv("MONGODB_URI", "mongodb://localhost:27017/mydb"))

# Configure your agent memory
agent_memory = ChromaAgentMemory(
    collection_name="mango_memory",
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

# Run the server
server = MangoFastAPIServer(agent)
server.run()  # http://localhost:8000
