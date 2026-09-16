"""Built-in access-control middlewares.

Each built-in takes a *function of the host's user object* so the role logic
stays entirely in the host application::

    from mango.middleware import CollectionAccess, RowFilter, RedactFields, DenyTools, Budget, AuditLog

    agent.use(CollectionAccess(lambda user: ROLE_COLLECTIONS[user.role]))
    agent.use(RowFilter("orders", lambda user: {} if user.is_admin else {"region": user.region}))
    agent.use(RedactFields(lambda user: [] if user.is_admin else ["customers.email"]))
    agent.use(DenyTools(lambda user: [] if user.is_admin else ["save_text_memory"]))
    agent.use(AuditLog())

    answer = await agent.ask(question, user=current_user)
"""

from mango.core.access import AccessDenied, FunctionMiddleware, Middleware, TurnContext
from mango.middleware.builtins import (
    AuditLog,
    Budget,
    CollectionAccess,
    DenyTools,
    RedactFields,
    RowFilter,
)

__all__ = [
    "AccessDenied",
    "Middleware",
    "FunctionMiddleware",
    "TurnContext",
    "CollectionAccess",
    "RowFilter",
    "RedactFields",
    "DenyTools",
    "Budget",
    "AuditLog",
]
