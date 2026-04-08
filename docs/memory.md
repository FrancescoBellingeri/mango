# Memory

Mango learns from every successful interaction. The more you use it, the faster and more accurate it gets.

---

## How It Works

Mango uses **ChromaDB** as its default vector store. After each successful tool call, it automatically saves the `(question, tool_name, args, result)` tuple as an embedding. On the next similar question, the saved example is retrieved and injected as a few-shot prompt for the LLM.

```
Question → embed → search memory
                        │
                        ├── similar past example found → inject as few-shot → faster, accurate answer
                        │
                        └── no match → LLM reasons from scratch → result auto-saved for next time
```

---

## Setup

```python
from mango.integrations.chromadb import ChromaAgentMemory

# Persistent (survives restarts)
memory = ChromaAgentMemory(
    collection_name="mango_memory",
    persist_dir="./mango_memory",
)

# In-memory only (useful for testing)
memory = ChromaAgentMemory(
    collection_name="mango_memory",
    persist_dir=":memory:",
)
```

---

## Pre-loading Domain Knowledge

Teach Mango about your business terminology before it starts answering questions:

```python
# Field semantics
memory.save_text("'active customer' means a customer who placed an order in the last 90 days")
memory.save_text("'revenue' always refers to the total_amount field in the orders collection")

# Enum values
memory.save_text("the 'status' field uses: 1=pending, 2=shipped, 3=delivered, 4=cancelled")

# Collection relationships
memory.save_text("orders reference customers via the customer_id field (ObjectId)")
memory.save_text("orderitems.order_id links to the orders collection")

# Business rules
memory.save_text("'top customers' are those with lifetime value > $10,000")
```

This is equivalent to the training data in Vanna — it gives the LLM the business context it needs to generate accurate queries from day one.

---

## Checking Memory Size

```python
print(f"Stored interactions: {memory.count()}")
```

---

## Multi-turn Memory

Memory persists across sessions. Questions asked today make tomorrow's answers better. There's no need to re-explain your database structure every time.

---

## Bring Your Own Backend

The `MemoryService` abstract class makes it easy to swap ChromaDB for any vector store:

```python
from mango.memory.base import MemoryService
from mango.memory.models import MemoryEntry, TextMemoryEntry

class MyPineconeMemory(MemoryService):
    async def store(self, entry: MemoryEntry) -> None:
        ...

    async def retrieve(self, question: str, top_k: int = 5) -> list[MemoryEntry]:
        ...

    async def save_text(self, text: str) -> str:
        ...

    async def search_text(self, query: str, top_k: int = 5) -> list[TextMemoryEntry]:
        ...

    async def delete(self, entry_id: str) -> None:
        ...

    def count(self) -> int:
        ...
```
