"""Mango — Mongo Agent for Natural-language Operations.

Quick start::

    from mango import MangoAgent
    from mango.core.registry import ToolRegistry
    from mango.tools import ListCollectionsTool, DescribeCollectionTool, RunMQLTool
    from mango.servers.fastapi import MangoFastAPIServer
    from mango.integrations.google import GeminiLlmService
    from mango.integrations.mongodb import MongoBackend
    from mango.integrations.chromadb import ChromaAgentMemory

    llm = GeminiLlmService(model="gemini-2.5-pro-preview-05-06", api_key="...")
    backend = MongoBackend()
    backend.connect("mongodb://localhost:27017/mydb")
    agent_memory = ChromaAgentMemory(persist_directory="./chroma_db")

    tools = ToolRegistry()
    tools.register(ListCollectionsTool(backend))
    tools.register(DescribeCollectionTool(backend))
    tools.register(RunMQLTool(backend))

    agent = MangoAgent(
        llm_service=llm,
        tool_registry=tools,
        backend=backend,
        agent_memory=agent_memory,
    )

    server = MangoFastAPIServer(agent)
    server.run()  # http://localhost:8000
"""

from mango.agent.agent import AgentResponse, MangoAgent

__all__ = ["MangoAgent", "AgentResponse"]
