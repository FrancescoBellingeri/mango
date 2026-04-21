"""
Agent memory module for Mango.

This module provides the base classes and models for agent memory.
"""

from .base import MemoryService
from .models import MemoryEntry, TextMemoryEntry, TrainingEntry, make_entry_id

__all__ = [
    "MemoryService",
    "MemoryEntry",
    "TextMemoryEntry",
    "TrainingEntry",
    "make_entry_id",
]