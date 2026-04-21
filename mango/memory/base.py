"""Abstract base class for agent memory.

Memory stores successful Q&A interactions and retrieves similar ones
to inject as few-shot examples into the LLM prompt.

All I/O methods are async so implementations can offload blocking work
(ChromaDB, network calls) without stalling FastAPI's event loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from mango.memory.models import MemoryEntry, TextMemoryEntry, TrainingEntry


class MemoryService(ABC):
    """ABC for memory backends.

    Implementations must support storing entries and retrieving the
    top-K most similar ones for a given query string.

    All I/O methods are coroutines — call them with ``await``.
    count() is the only synchronous method (trivially fast).
    """

    # ------------------------------------------------------------------
    # Tool-usage memory
    # ------------------------------------------------------------------

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Persist a successful interaction.

        Args:
            entry: The interaction to store. entry.id must be unique.
        """

    @abstractmethod
    async def retrieve(
        self,
        question: str,
        top_k: int = 3,
        similarity_threshold: float = 0.6,
    ) -> list[MemoryEntry]:
        """Return the top_k most similar stored interactions.

        Args:
            question: The current user question used as search query.
            top_k: Max number of results to return.
            similarity_threshold: Minimum similarity score (0-1) to include.

        Returns:
            List of MemoryEntry sorted by similarity (most similar first),
            with entry.similarity populated.
        """

    @abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Remove a stored entry by ID.

        Args:
            entry_id: The ID of the entry to remove.
        """

    # ------------------------------------------------------------------
    # Text memory (glossary, domain notes)
    # ------------------------------------------------------------------

    @abstractmethod
    async def save_text(self, text: str) -> str:
        """Store a free-form text memory (business glossary, domain notes).

        Args:
            text: The text content to store.

        Returns:
            The generated entry ID.
        """

    @abstractmethod
    async def search_text(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.6,
    ) -> list[TextMemoryEntry]:
        """Retrieve text memories semantically similar to query.

        Args:
            query: The search query.
            top_k: Max number of results to return.
            similarity_threshold: Minimum similarity score (0-1) to include.

        Returns:
            List of TextMemoryEntry sorted by similarity.
        """

    # ------------------------------------------------------------------
    # Training (gold-standard examples)
    # ------------------------------------------------------------------

    @abstractmethod
    async def train(self, entry: TrainingEntry) -> None:
        """Persist a verified training entry.

        Training entries are gold-standard examples loaded by the user.
        They are never overwritten by the auto-save logic.
        """

    @abstractmethod
    async def get_training_entries(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> list[TrainingEntry]:
        """Return the top_k training entries most similar to question."""

    @abstractmethod
    def training_count(self) -> int:
        """Return the total number of stored training entries."""

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    @abstractmethod
    async def export_all(self) -> list[dict]:
        """Export all stored entries (tool-usage, text, training) as plain dicts.

        Each dict has a ``"type"`` key: ``"tool"``, ``"text"``, or ``"training"``.
        Suitable for serialisation to JSONL.
        """

    @abstractmethod
    async def import_all(self, entries: list[dict]) -> int:
        """Import entries exported by export_all().

        Returns:
            Number of entries successfully imported.
        """

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored tool-usage entries."""
