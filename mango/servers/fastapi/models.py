"""Pydantic request and response models for the Mango REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Upper bounds on client-supplied text. A question is a sentence or two; a
# multi-kilobyte payload is either a mistake or an attempt to run up the LLM
# bill / smuggle instructions into the prompt.
MAX_QUESTION_CHARS = 4_000
MAX_SESSION_ID_CHARS = 64
MAX_IMPORT_ENTRIES = 1_000


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=MAX_QUESTION_CHARS,
        description="Natural language question",
    )
    session_id: str | None = Field(
        default=None,
        max_length=MAX_SESSION_ID_CHARS,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Session ID to continue an existing conversation. "
                    "Omit to start a new one.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str


class TrainRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    tool_name: str = Field(..., min_length=1, max_length=100)
    tool_args: dict = Field(default_factory=dict)
    result_summary: str = Field(default="", max_length=MAX_QUESTION_CHARS)


class TrainResponse(BaseModel):
    imported: int


class ExportResponse(BaseModel):
    entries: list[dict]
    count: int
