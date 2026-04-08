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
from mango.integrations.google import GeminiLlmService
from mango.integrations.mongodb import MongoRunner
from mango.integrations.chromadb import ChromaAgentMemory
from dotenv import load_dotenv

load_dotenv()

# Configure your LLM
llm = GeminiLlmService(
    model="gemini-3.1-pro-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
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
tools.register_local_tool(ListCollectionsTool(db))
tools.register_local_tool(SearchCollectionsTool(db))
tools.register_local_tool(DescribeCollectionTool(db))
tools.register_local_tool(CollectionStatsTool(db))
tools.register_local_tool(RunMQLTool(db))
tools.register_local_tool(SearchSavedCorrectToolUsesTool(agent_memory))
tools.register_local_tool(SaveTextMemoryTool(agent_memory))

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
