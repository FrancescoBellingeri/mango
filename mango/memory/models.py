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


@dataclass
class TextMemoryEntry:
    """A stored free-form text memory (glossary, domain notes, etc.)."""

    id: str
    text: str
    similarity: float = 0.0  # filled in by search_text()


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