from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


Side = Literal["pro", "con"]
Winner = Literal["pro", "con"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DebateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=2000)
    question: str = Field(min_length=1, max_length=4000)
    context: str = Field(default="", max_length=6000)
    max_rounds: int = Field(default=12, ge=1, le=12)

    @field_validator("topic", "question", "context")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())


class AgentTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=12000)
    claims: list[str] = Field(default_factory=list, max_length=8)
    evidence: list[str] = Field(default_factory=list, max_length=8)
    rebuttals: list[str] = Field(default_factory=list, max_length=8)


class DebateMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debate_id: UUID
    round_number: int = Field(ge=1, le=12)
    side: Side
    content: str = Field(min_length=1, max_length=12000)
    claims: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Winner
    pro_score: int = Field(ge=0, le=100)
    con_score: int = Field(ge=0, le=100)
    reasoning: str = Field(min_length=1, max_length=12000)
    pro_strengths: list[str] = Field(default_factory=list, max_length=8)
    con_strengths: list[str] = Field(default_factory=list, max_length=8)
    pro_weaknesses: list[str] = Field(default_factory=list, max_length=8)
    con_weaknesses: list[str] = Field(default_factory=list, max_length=8)
    final_summary: str = Field(min_length=1, max_length=12000)


class DebateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debate_id: UUID
    input: DebateInput
    messages: list[DebateMessage]
    judgment: JudgeResult
    completed_at: datetime = Field(default_factory=utc_now)

