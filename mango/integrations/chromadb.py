"""ChromaDB implementation of MemoryService.

Uses ChromaDB's built-in embedding model (all-MiniLM-L6-v2 via sentence-transformers)
to embed questions and retrieve semantically similar ones.

Storage is persistent on disk by default, so memory survives restarts.
All I/O methods are async — ChromaDB (which is synchronous) is executed
via asyncio.to_thread() to avoid blocking the event loop.

Two separate ChromaDB collections are used:
  - ``<name>``        for tool-usage memories (question → tool_args pairs)
  - ``<name>_text``   for free-form text memories (glossary, domain notes)
"""

from __future__ import annotations

import asyncio
import json
import logging

import chromadb
from chromadb.config import Settings

from mango.memory import MemoryEntry, MemoryService, TextMemoryEntry, TrainingEntry, make_entry_id

logger = logging.getLogger(__name__)


class ChromaAgentMemory(MemoryService):
    """MemoryService backed by ChromaDB.

    Args:
        persist_dir: Directory where ChromaDB stores its data.
            Defaults to '.mango_memory' in the current directory.
            Pass ':memory:' to use an in-memory (non-persistent) instance.
        collection_name: Base name for the ChromaDB collections.
            Tool memories go into ``<collection_name>``,
            text memories into ``<collection_name>_text``.
    """

    @staticmethod
    def _get_or_create(client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
        # get_collection first to avoid re-instantiating the embedding model on restart.
        try:
            return client.get_collection(name=name)
        except Exception:
            return client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    def __init__(
        self,
        persist_dir: str = ".mango_memory",
        collection_name: str = "mango_memory",
    ) -> None:
        if persist_dir == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._get_or_create(self._client, collection_name)
        self._text_collection = self._get_or_create(self._client, f"{collection_name}_text")
        self._training_collection = self._get_or_create(self._client, f"{collection_name}_training")

        logger.info(
            "ChromaMemoryService ready — tool entries: %d, text entries: %d, training entries: %d.",
            self._collection.count(),
            self._text_collection.count(),
            self._training_collection.count(),
        )

    # ------------------------------------------------------------------
    # Tool-usage memory
    # ------------------------------------------------------------------

    async def store(self, entry: MemoryEntry) -> None:
        """Store a MemoryEntry. Silently overwrites if the ID already exists."""
        def _sync() -> None:
            metadata = {
                "tool_name": entry.tool_name,
                "tool_args": json.dumps(entry.tool_args, default=str),
                "result_summary": entry.result_summary,
            }
            self._collection.upsert(
                ids=[entry.id],
                documents=[entry.question],
                metadatas=[metadata],
            )
            logger.debug("Stored tool memory '%s': %s", entry.id, entry.question)

        await asyncio.to_thread(_sync)

    async def retrieve(
        self,
        question: str,
        top_k: int = 3,
        similarity_threshold: float = 0.6,
    ) -> list[MemoryEntry]:
        """Return top_k semantically similar stored tool-usage entries."""
        def _sync() -> list[MemoryEntry]:
            total = self._collection.count()
            if total == 0:
                return []

            n = min(top_k, total)
            results = self._collection.query(
                query_texts=[question],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )

            entries: list[MemoryEntry] = []
            for i, entry_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = max(0.0, 1.0 - distance)

                if similarity < similarity_threshold:
                    continue

                try:
                    tool_args = json.loads(meta.get("tool_args", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                entries.append(
                    MemoryEntry(
                        id=entry_id,
                        question=results["documents"][0][i],
                        tool_name=meta.get("tool_name", ""),
                        tool_args=tool_args,
                        result_summary=meta.get("result_summary", ""),
                        similarity=round(similarity, 3),
                    )
                )
            return entries

        return await asyncio.to_thread(_sync)

    async def delete(self, entry_id: str) -> None:
        """Delete a stored tool entry by ID. No-op if not found."""
        def _sync() -> None:
            try:
                self._collection.delete(ids=[entry_id])
                logger.debug("Deleted tool memory '%s'.", entry_id)
            except Exception as exc:
                logger.warning("Could not delete tool memory '%s': %s", entry_id, exc)

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Text memory
    # ------------------------------------------------------------------

    async def save_text(self, text: str) -> str:
        """Store a free-form text memory. Returns the generated entry ID."""
        entry_id = make_entry_id()

        def _sync() -> None:
            self._text_collection.upsert(
                ids=[entry_id],
                documents=[text],
                metadatas=[{"type": "text"}],
            )
            logger.debug("Stored text memory '%s'.", entry_id)

        await asyncio.to_thread(_sync)
        return entry_id

    async def search_text(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.6,
    ) -> list[TextMemoryEntry]:
        """Retrieve text memories semantically similar to query."""
        def _sync() -> list[TextMemoryEntry]:
            total = self._text_collection.count()
            if total == 0:
                return []

            n = min(top_k, total)
            results = self._text_collection.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "distances"],
            )

            entries: list[TextMemoryEntry] = []
            for i, entry_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = max(0.0, 1.0 - distance)
                if similarity < similarity_threshold:
                    continue
                entries.append(
                    TextMemoryEntry(
                        id=entry_id,
                        text=results["documents"][0][i],
                        similarity=round(similarity, 3),
                    )
                )
            return entries

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Training (gold-standard examples)
    # ------------------------------------------------------------------

    async def train(self, entry: TrainingEntry) -> None:
        """Store a verified training entry. Silently overwrites if ID already exists."""
        def _sync() -> None:
            metadata = {
                "tool_name": entry.tool_name,
                "tool_args": json.dumps(entry.tool_args, default=str),
                "result_summary": entry.result_summary,
                "timestamp": entry.timestamp,
            }
            self._training_collection.upsert(
                ids=[entry.id],
                documents=[entry.question],
                metadatas=[metadata],
            )
            logger.debug("Stored training entry '%s': %s", entry.id, entry.question)

        await asyncio.to_thread(_sync)

    async def get_training_entries(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> list[TrainingEntry]:
        """Return top_k training entries semantically similar to question."""
        def _sync() -> list[TrainingEntry]:
            total = self._training_collection.count()
            if total == 0:
                return []

            n = min(top_k, total)
            results = self._training_collection.query(
                query_texts=[question],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )

            entries: list[TrainingEntry] = []
            for i, entry_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = max(0.0, 1.0 - distance)

                if similarity < similarity_threshold:
                    continue

                try:
                    tool_args = json.loads(meta.get("tool_args", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                entries.append(
                    TrainingEntry(
                        id=entry_id,
                        question=results["documents"][0][i],
                        tool_name=meta.get("tool_name", ""),
                        tool_args=tool_args,
                        result_summary=meta.get("result_summary", ""),
                        similarity=round(similarity, 3),
                        timestamp=meta.get("timestamp", ""),
                    )
                )
            return entries

        return await asyncio.to_thread(_sync)

    def training_count(self) -> int:
        """Total number of stored training entries."""
        return self._training_collection.count()

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    async def export_all(self) -> list[dict]:
        """Export all entries (tool-usage, text, training) as plain dicts."""
        def _sync() -> list[dict]:
            out: list[dict] = []

            # Tool-usage entries
            total = self._collection.count()
            if total > 0:
                results = self._collection.get(include=["documents", "metadatas"])
                for i, entry_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i]
                    try:
                        tool_args = json.loads(meta.get("tool_args", "{}"))
                    except json.JSONDecodeError:
                        tool_args = {}
                    out.append({
                        "type": "tool",
                        "id": entry_id,
                        "question": results["documents"][i],
                        "tool_name": meta.get("tool_name", ""),
                        "tool_args": tool_args,
                        "result_summary": meta.get("result_summary", ""),
                    })

            # Text entries
            text_total = self._text_collection.count()
            if text_total > 0:
                results = self._text_collection.get(include=["documents"])
                for i, entry_id in enumerate(results["ids"]):
                    out.append({
                        "type": "text",
                        "id": entry_id,
                        "text": results["documents"][i],
                    })

            # Training entries
            training_total = self._training_collection.count()
            if training_total > 0:
                results = self._training_collection.get(include=["documents", "metadatas"])
                for i, entry_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i]
                    try:
                        tool_args = json.loads(meta.get("tool_args", "{}"))
                    except json.JSONDecodeError:
                        tool_args = {}
                    out.append({
                        "type": "training",
                        "id": entry_id,
                        "question": results["documents"][i],
                        "tool_name": meta.get("tool_name", ""),
                        "tool_args": tool_args,
                        "result_summary": meta.get("result_summary", ""),
                        "timestamp": meta.get("timestamp", ""),
                    })

            return out

        return await asyncio.to_thread(_sync)

    async def import_all(self, entries: list[dict]) -> int:
        """Import entries exported by export_all(). Returns count imported."""
        imported = 0
        for raw in entries:
            entry_type = raw.get("type")
            try:
                if entry_type == "tool":
                    await self.store(MemoryEntry(
                        id=raw.get("id", make_entry_id()),
                        question=raw["question"],
                        tool_name=raw["tool_name"],
                        tool_args=raw.get("tool_args", {}),
                        result_summary=raw.get("result_summary", ""),
                    ))
                    imported += 1
                elif entry_type == "text":
                    def _upsert_text(eid: str, text: str) -> None:
                        self._text_collection.upsert(
                            ids=[eid],
                            documents=[text],
                            metadatas=[{"type": "text"}],
                        )
                    await asyncio.to_thread(_upsert_text, raw.get("id", make_entry_id()), raw["text"])
                    imported += 1
                elif entry_type == "training":
                    await self.train(TrainingEntry(
                        id=raw.get("id", make_entry_id()),
                        question=raw["question"],
                        tool_name=raw["tool_name"],
                        tool_args=raw.get("tool_args", {}),
                        result_summary=raw.get("result_summary", ""),
                        timestamp=raw.get("timestamp", ""),
                    ))
                    imported += 1
                else:
                    logger.warning("Unknown entry type '%s' during import — skipped.", entry_type)
            except Exception as exc:
                logger.warning("Failed to import entry: %s", exc)

        return imported

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of stored tool-usage entries."""
        return self._collection.count()


