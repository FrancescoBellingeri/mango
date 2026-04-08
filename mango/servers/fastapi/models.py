"""Pydantic request and response models for the Mango REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    session_id: str | None = Field(
        default=None,
        description="Session ID to continue an existing conversation. "
                    "Omit to start a new one.",
    )

class HealthResponse(BaseModel):
    status: str
    service: str
