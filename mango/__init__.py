"""Mango — Mongo Agent for Natural-language Operations.

Quick start::

    from mango import MangoAgent
    from mango.tools import ToolRegistry, build_mongo_tools, SaveTextMemoryTool
    from mango.integrations.anthropic import AnthropicLlmService
    from mango.integrations.mongodb import MongoRunner
    from mango.integrations.chromadb import ChromaAgentMemory
    from mango.servers.fastapi import MangoFastAPIServer

    llm = AnthropicLlmService(model="claude-sonnet-4-6", api_key="...")
    db = MongoRunner()
    db.connect("mongodb://localhost:27017/mydb")
    agent_memory = ChromaAgentMemory(persist_dir="./chroma_db")

    tools = ToolRegistry()
    for tool in build_mongo_tools(db):
        tools.register(tool)
    tools.register(SaveTextMemoryTool(agent_memory))

    agent = MangoAgent(
        llm_service=llm,
        tool_registry=tools,
        db=db,
        agent_memory=agent_memory,
        introspect=True,
    )

    # Per-user permissions: see mango.middleware
    server = MangoFastAPIServer(agent, user_for=lambda request: request.state.user)
    server.run()  # http://localhost:8000
"""

from mango.agent.agent import AgentResponse, MangoAgent

__version__ = "0.2.0"

__all__ = ["MangoAgent", "AgentResponse", "__version__"]
