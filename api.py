from __future__ import annotations

import io
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.analytics_store import AnalyticsStore
from backend.config import ANALYTICS_DB_PATH
from backend.schemas import AnalyzeRequest, AnalyzeResponse, SummaryResponse
from backend.service import NewsAnalyzerService
from backend.text_validation import TextValidationError


app = FastAPI(
    title="AI Fake News Analyzer API",
    version="2.0.0",
    description="ML + XAI API for fake-news detection with analytics export.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = NewsAnalyzerService()
store = AnalyticsStore(ANALYTICS_DB_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = analyzer.analyze(text=req.text, language=req.language)
    except TextValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.result.to_detail()) from exc

    if req.store_event:
        store.log_check(
            language=result.detected_language,
            content_type=req.content_type,
            country=req.country,
            label=result.label,
            fake_prob=result.fake_probability,
            real_prob=result.real_probability,
            text_length=len(req.text),
            token_length=result.token_length,
            latency_ms=result.latency_ms,
            model_trace=result.model_trace,
        )

    return AnalyzeResponse(
        label=result.label,
        fake_probability=result.fake_probability,
        real_probability=result.real_probability,
        detected_language=result.detected_language,  # type: ignore[arg-type]
        translated_to_english=result.translated_to_english,
        translation_text=result.translation_text,
        latency_ms=result.latency_ms,
        explanation=result.explanation,  # type: ignore[arg-type]
        model_trace=result.model_trace,
    )


def _parse_multi(value: str | None) -> List[str]:
    if not value:
        return []
    items = [x.strip() for x in value.split(",") if x.strip()]
    return sorted(set(items))


@app.get("/analytics/summary", response_model=SummaryResponse)
def analytics_summary(
    days: int = Query(30, ge=0, le=3650),
    languages: str | None = Query(None, description="Comma-separated, e.g. en,ru"),
    content_types: str | None = Query(None, description="Comma-separated, e.g. article,post"),
) -> SummaryResponse:
    summ = store.summary(
        days=days,
        languages=_parse_multi(languages),
        content_types=_parse_multi(content_types),
    )
    return SummaryResponse(
        total_checks=summ.total_checks,
        fake_checks=summ.fake_checks,
        real_checks=summ.real_checks,
        fake_share=summ.fake_share,
        avg_latency_ms=summ.avg_latency_ms,
    )


@app.get("/analytics/timeline")
def analytics_timeline(
    days: int = Query(30, ge=0, le=3650),
    languages: str | None = None,
    content_types: str | None = None,
) -> list[dict]:
    df = store.timeline(
        days=days,
        languages=_parse_multi(languages),
        content_types=_parse_multi(content_types),
    )
    return df.to_dict(orient="records")


@app.get("/analytics/languages")
def analytics_languages(
    days: int = Query(30, ge=0, le=3650),
    content_types: str | None = None,
) -> list[dict]:
    df = store.language_distribution(days=days, content_types=_parse_multi(content_types))
    return df.to_dict(orient="records")


@app.get("/analytics/geo")
def analytics_geo(
    days: int = Query(30, ge=0, le=3650),
    languages: str | None = None,
    content_types: str | None = None,
) -> list[dict]:
    df = store.geo_distribution(
        days=days,
        languages=_parse_multi(languages),
        content_types=_parse_multi(content_types),
    )
    return df.to_dict(orient="records")


@app.get("/analytics/export")
def analytics_export(
    fmt: str = Query("csv", pattern="^(csv|json)$"),
    days: int = Query(30, ge=0, le=3650),
    languages: str | None = None,
    content_types: str | None = None,
):
    df = store.export_checks(
        days=days,
        languages=_parse_multi(languages),
        content_types=_parse_multi(content_types),
    )
    if fmt == "json":
        return JSONResponse(content=df.to_dict(orient="records"))

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    bytes_io = io.BytesIO(buffer.getvalue().encode("utf-8"))
    return StreamingResponse(
        bytes_io,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analytics_export.csv"},
    )
