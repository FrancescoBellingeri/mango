"""Mango FastAPI server.

Two ways to use it:

- standalone: ``MangoFastAPIServer(agent, user_for=...).run()``
- inside your app: ``app.include_router(mango_router(agent, user_for=...), prefix="/mango")``
"""

from mango.servers.fastapi.main import MangoFastAPIServer
from mango.servers.fastapi.routes import mango_router

__all__ = ["MangoFastAPIServer", "mango_router"]
