from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Language = Literal["auto", "en", "ru", "kk"]
ContentType = Literal["article", "post", "headline", "other"]


class Highlight(BaseModel):
    fragment: str
    reason: str


class ExplainItem(BaseModel):
    type: Literal["explicit", "implicit", "context"]
    text: str
    weight: float = Field(ge=0.0, le=1.0)
    code: str


class ExplainPayload(BaseModel):
    probability_fake_percent: float = Field(ge=0.0, le=100.0)
    reasons: List[ExplainItem]
    highlights: List[Highlight]
    compact_codes: List[str]
    agent_hint_min_tokens: Optional[str] = None
    detailed_summary: Optional[str] = None
    decision_path: List[str] = Field(default_factory=list)
    feature_snapshot: Dict[str, float] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=5, max_length=50000)
    language: Language = "auto"
    content_type: ContentType = "article"
    country: Optional[str] = None
    store_event: bool = True


class AnalyzeResponse(BaseModel):
    label: Literal["fake", "real"]
    fake_probability: float = Field(ge=0.0, le=1.0)
    real_probability: float = Field(ge=0.0, le=1.0)
    detected_language: Literal["en", "ru", "kk"]
    translated_to_english: bool
    translation_text: Optional[str] = None
    latency_ms: int = Field(ge=0)
    explanation: ExplainPayload
    model_trace: str


class SummaryResponse(BaseModel):
    total_checks: int
    fake_checks: int
    real_checks: int
    fake_share: float
    avg_latency_ms: float


class Point(BaseModel):
    bucket: str
    value: int
