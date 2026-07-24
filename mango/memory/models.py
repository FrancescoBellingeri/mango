"""Memory models for Mango.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime


def make_entry_id() -> str:
    """Generate a unique ID for a new MemoryEntry."""
    return str(uuid.uuid4())


@dataclass
class MemoryEntry:
    """A stored successful tool-usage interaction."""

    id: str
    question: str           # original natural language question
    tool_name: str          # tool that produced the answer, e.g. 'run_mql'
    tool_args: dict         # exact args that worked
    result_summary: str     # brief human-readable summary of the result
    similarity: float = 0.0 # filled in by retrieve(), distance score
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Provenance for text memories. Legacy records without metadata map to "legacy".
TEXT_MEMORY_SOURCES = frozenset({"manual", "import", "llm", "legacy"})


@dataclass
class TextMemoryEntry:
    """A stored free-form text memory (glossary, domain notes, etc.).

    Provenance policy (P1 / §5):
      - ``manual`` / ``import``: human- or API-supplied; ``verified=True`` by default.
      - ``llm``: written by the agent via ``save_text_memory``; never treated as verified.
      - ``legacy``: pre-provenance records; conservative — ``verified=False``.

    Retrieval always frames notes as non-authoritative reference data regardless
    of ``verified``. Callers may exclude unverified notes via agent config.
    """

    id: str
    text: str
    similarity: float = 0.0  # filled in by search_text() — retrieval score, not confidence
    source: str = "legacy"
    verified: bool = False


@dataclass
class TrainingEntry:
    """A verified (question → tool_args) pair used as gold-standard few-shot example.

    Unlike MemoryEntry (auto-saved), TrainingEntry is explicitly loaded by the
    user via /train and never overwritten by auto-save logic.
    """

    id: str
    question: str
    tool_name: str
    tool_args: dict
    result_summary: str = ""
    similarity: float = 0.0  # filled in by get_training_entries()
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
