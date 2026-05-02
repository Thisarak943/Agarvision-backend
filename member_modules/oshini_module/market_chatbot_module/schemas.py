from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(default="default")


class ChatResponse(BaseModel):
    intent: str
    language: str
    reply: str
    status: str = "success"
    missing_fields: list[str] = []
    extracted: dict[str, Any] = {}
    prediction: dict[str, Any] | None = None


class ResetRequest(BaseModel):
    session_id: str | None = Field(default="default")
