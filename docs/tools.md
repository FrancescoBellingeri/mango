# Tools

Mango comes with a set of built-in tools that the agent can call to interact with MongoDB and its memory. All tools are read-only by design.

---

## Built-in Tools

### MongoDB Tools

| Tool | Name | Description |
|------|------|-------------|
| `ListCollectionsTool` | `list_collections` | Lists all collections. Databases with 100+ collections are grouped by name prefix automatically. |
| `SearchCollectionsTool` | `search_collections` | Searches collections by name pattern (supports glob: `order*`, `*_log`). |
| `DescribeCollectionTool` | `describe_collection` | Returns full schema for a collection: field types, frequencies, indexes, and inferred references. |
| `CollectionStatsTool` | `collection_stats` | Returns document count and storage size for a collection. |
| `RunMQLTool` | `run_mql` | Executes a read-only MongoDB query. Accepts `find`, `aggregate`, `count`, `distinct`. Write operations are rejected. |

### Memory Tools

| Tool | Name | Description |
|------|------|-------------|
| `SearchSavedCorrectToolUsesTool` | `search_saved_correct_tool_uses` | Searches memory for similar past interactions (few-shot examples). |
| `SaveTextMemoryTool` | `save_text_memory` | Saves free-form knowledge about the database for future queries. |

---

## Registering Tools

```python
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

tools = ToolRegistry()
tools.register(ListCollectionsTool(db))
tools.register(SearchCollectionsTool(db))
tools.register(DescribeCollectionTool(db))
tools.register(CollectionStatsTool(db))
tools.register(RunMQLTool(db))
tools.register(SearchSavedCorrectToolUsesTool(memory))
tools.register(SaveTextMemoryTool(memory))
```

Or use the shortcut that registers all MongoDB tools at once:

```python
from mango.tools import build_mongo_tools

for tool in build_mongo_tools(db):
    tools.register(tool)
```

---

## Building a Custom Tool

Extend `Tool` and implement `definition` and `async execute`:

```python
from mango.tools.base import Tool, ToolResult
from mango.llm.models import ToolDef, ToolParam

class PriceLookupTool(Tool):
    """Looks up the current price for a product SKU from an external API."""

    def __init__(self, price_api_client):
        self._api = price_api_client

    @property
    def definition(self) -> ToolDef:
        return ToolDef(
            name="price_lookup",
            description="Get the current price for a product SKU",
            params=[
                ToolParam(
                    name="sku",
                    type="string",
                    description="The product SKU to look up",
                    required=True,
                )
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        sku = kwargs["sku"]
        try:
            price = await self._api.get_price(sku)
            return ToolResult(success=True, output=f"Price for {sku}: ${price:.2f}")
        except Exception as e:
            return ToolResult(success=False, output=str(e))

# Register it alongside the built-in tools
tools.register(PriceLookupTool(my_api_client))
```

The agent will automatically discover and use your tool when relevant.

---

## Read-only Enforcement

`RunMQLTool` only accepts the following MongoDB operations:

- `find`
- `aggregate`
- `count`
- `distinct`

Any attempt to run `insertOne`, `updateMany`, `deleteOne`, or any write operation is rejected at the tool level before reaching MongoDB.
