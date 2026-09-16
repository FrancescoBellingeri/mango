"""Complete example: Mango server on mango_marketplace with per-role access control.

Run it, then talk to it with different users (see the curl block at the bottom).
The host application is responsible for authentication: here it is a header
``X-User: <name>`` looked up in a demo table — replace ``user_for`` with your
session/JWT check.
"""

import asyncio
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import HTTPException, Request

from mango import MangoAgent
from mango.integrations import (
    AnthropicLlmService,
    ChromaAgentMemory,
    GeminiLlmService,
    MongoRunner,
    OllamaLlmService,
    OpenAILlmService,
)
from mango.middleware import (
    AccessDenied,
    AuditLog,
    Budget,
    CollectionAccess,
    DenyTools,
    RedactFields,
    RowFilter,
)
from mango.servers.fastapi import MangoFastAPIServer
from mango.tools import (
    CollectionStatsTool,
    DeleteLastMemoryEntryTool,
    DescribeCollectionTool,
    InspectFieldTool,
    ListCollectionsTool,
    RunMQLTool,
    SaveTextMemoryTool,
    SearchCollectionsTool,
    ToolRegistry,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# 1. Your users and roles — this is YOUR domain, Mango never looks inside.
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: str
    role: str            # admin | support | finance
    country: str = ""    # support agents only see customers of their country


USERS = {
    "alice": User(id="alice", role="admin"),
    "bob": User(id="bob", role="support", country="IT"),
    "carol": User(id="carol", role="finance"),
}

ROLE_COLLECTIONS = {
    "admin": "*",
    "support": ["users", "orders", "returns", "shipments", "support_tickets"],
    "finance": ["orders", "payments", "ledger_entries", "merchants", "currencies", "fx_rates"],
}

# Fields hidden from everyone but admins (PII + payment details).
HIDDEN_FIELDS = ["users.email", "users.phone", "users.addresses", "payments.details"]


def user_for(request: Request) -> User:
    """Your authentication. Here: a header; in real life: session, JWT, SSO."""
    name = request.headers.get("X-User", "")
    user = USERS.get(name)
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user (set X-User: alice|bob|carol)")
    return user


# ---------------------------------------------------------------------------
# 2. LLM / database / memory / tools — as before
# ---------------------------------------------------------------------------

llm = OllamaLlmService(
    model="gemma4:12b-mlx",
    # api_key=os.getenv("ANTHROPIC_API_KEY"),
)

db = MongoRunner()
db.connect(os.getenv("MONGODB_URI", "mongodb://localhost:27017/mango_marketplace"))

agent_memory = ChromaAgentMemory(persist_dir=".mango_memory")

tools = ToolRegistry()
tools.register(ListCollectionsTool(db))
tools.register(SearchCollectionsTool(db))
tools.register(DescribeCollectionTool(db))
tools.register(InspectFieldTool(db))
tools.register(CollectionStatsTool(db))
tools.register(RunMQLTool(db))
tools.register(SaveTextMemoryTool(agent_memory))

agent = MangoAgent(
    llm_service=llm,
    tool_registry=tools,
    db=db,
    agent_memory=agent_memory,
    introspect=True,   # schema in the prompt, filtered per user by the middlewares
)
tools.register(DeleteLastMemoryEntryTool(agent_memory, lambda: agent._last_memory_entry_id))

# ---------------------------------------------------------------------------
# 3. Access control: what each role may see and do
# ---------------------------------------------------------------------------

agent.use(CollectionAccess(lambda u: ROLE_COLLECTIONS[u.role]))
agent.use(RowFilter("users", lambda u: {} if u.role == "admin" else {"country": u.country}))
agent.use(RedactFields(lambda u: [] if u.role == "admin" else HIDDEN_FIELDS))
agent.use(DenyTools(lambda u: [] if u.role == "admin" else [
    "save_text_memory", "delete_last_memory_entry",   # tools
    "memory_train", "memory_import", "memory_export",  # REST endpoints
]))
agent.use(Budget(max_turns_per_day=200, max_queries_per_day=1000))
agent.use(AuditLog())   # JSON lines on the "mango.audit" logger


@agent.before_query
def finance_cannot_list_individual_payments(ctx, query):
    """A custom rule: finance may aggregate payments, not browse them one by one."""
    if ctx.user.role == "finance" and query.collection == "payments" and query.operation == "find":
        raise AccessDenied("finance may only aggregate payments, not list individual ones")
    return query


# ---------------------------------------------------------------------------
# 4. Serve
# ---------------------------------------------------------------------------

server = MangoFastAPIServer(agent, user_for=user_for)
server.run()  # http://localhost:8000

# ---------------------------------------------------------------------------
# Try it (each line streams SSE; -N keeps the connection open):
#
#   ask() { curl -sN -X POST localhost:8000/api/v1/ask/stream -H "X-User: $1" \
#           -H "Content-Type: application/json" -d "{\"question\": \"$2\"}"; }
#
#   ask bob   "quanti utenti ci sono?"                 # only country=IT (RowFilter)
#   ask alice "quanti utenti ci sono?"                 # all of them
#   ask bob   "dammi email e telefono di 3 utenti"     # denied (RedactFields)
#   ask bob   "quante righe ci sono in ledger_entries?" # denied (CollectionAccess)
#   ask carol "totale pagamenti per metodo"            # ok (aggregate)
#   ask carol "mostrami 5 pagamenti"                   # denied by the custom rule
#   ask bob   "ricorda che gli utenti IT sono i migliori"  # save_text_memory denied
#   curl -s localhost:8000/api/v1/memory/export -H "X-User: bob"    # 403
#   curl -s localhost:8000/api/v1/memory/export -H "X-User: alice"  # 200
#   curl -s localhost:8000/api/v1/health                             # no user needed
# ---------------------------------------------------------------------------
