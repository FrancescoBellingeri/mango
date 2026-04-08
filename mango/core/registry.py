"""ToolRegistry — public canonical import path.

The implementation lives in mango.tools.base to avoid circular imports.
Import from here in user scripts::

    from mango.core.registry import ToolRegistry
"""

from mango.tools.base import ToolRegistry

__all__ = ["ToolRegistry"]
