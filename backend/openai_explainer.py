from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .config import (
    OPENAI_API_KEY,
    OPENAI_EXPLANATION_ENABLED,
    OPENAI_EXPLANATION_MODEL,
    OPENAI_EXPLANATION_REASONING,
    OPENAI_EXPLANATION_TIMEOUT,
)


@dataclass
class OpenAIExplanation:
    text: Optional[str]
    model: Optional[str]
    latency_ms: int
    error: Optional[str] = None


def _extract_text(payload: Dict[str, Any]) -> Optional[str]:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
                continue
            alt_text = content.get("output_text")
            if isinstance(alt_text, str) and alt_text.strip():
                chunks.append(alt_text.strip())

    joined = "\n\n".join(chunk for chunk in chunks if chunk)
    return joined.strip() or None


def generate_openai_explanation(
    *,
    original_text: str,
    detected_language: str,
    label: str,
    fake_probability: float,
    real_probability: float,
    xai_summary: str,
    decision_path: list[str],
    highlights: list[dict[str, str]],
    reasons: list[dict[str, Any]],
) -> OpenAIExplanation:
    if not OPENAI_EXPLANATION_ENABLED:
        return OpenAIExplanation(
            text=None,
            model=None,
            latency_ms=0,
            error="LLM-обоснование выключено в конфиге.",
        )
    if not OPENAI_API_KEY:
        return OpenAIExplanation(
            text=None,
            model=None,
            latency_ms=0,
            error="OPENAI_API_KEY не задан в .env, поэтому LLM-обоснование пропущено.",
        )

    excerpt = (original_text or "")[:2200]
    highlights_text = ", ".join(
        f"{item.get('fragment', '')}: {item.get('reason', '')}" for item in highlights[:5]
    ) or "нет"
    reasons_text = "\n".join(
        f"- [{item.get('code', '-')}] {item.get('text', '')} (weight={item.get('weight', 0)})"
        for item in reasons[:6]
    ) or "- нет"
    decision_text = "\n".join(f"- {step}" for step in decision_path[:6]) or "- нет"

    instructions = (
        "Ты объясняешь результат уже выполненного классификатора фейк-новостей. "
        "Не придумывай внешние факты и не делай вид, что ты проверял интернет или первоисточники. "
        "Опирайся только на предоставленный текст, вероятности модели и XAI-сигналы. "
        "Пиши по-русски, 1 короткий абзац на 4-6 предложений. "
        "Тон: аккуратный, понятный, без категоричности. "
        "Если уверенность ограничена, честно скажи об этом. "
        "Если входной текст выглядит неполным, упомяни это."
    )

    user_prompt = (
        f"Вердикт модели: {label}\n"
        f"Вероятность fake: {fake_probability:.3f}\n"
        f"Вероятность real: {real_probability:.3f}\n"
        f"Определенный язык: {detected_language}\n"
        f"Краткий XAI-итог: {xai_summary}\n\n"
        f"Причины:\n{reasons_text}\n\n"
        f"Путь решения:\n{decision_text}\n\n"
        f"Подсвеченные фрагменты:\n{highlights_text}\n\n"
        f"Фрагмент текста:\n{excerpt}"
    )

    payload: Dict[str, Any] = {
        "model": OPENAI_EXPLANATION_MODEL,
        "instructions": instructions,
        "input": user_prompt,
        "max_output_tokens": 260,
        "store": False,
    }
    if OPENAI_EXPLANATION_REASONING != "none":
        payload["reasoning"] = {"effort": OPENAI_EXPLANATION_REASONING}

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=OPENAI_EXPLANATION_TIMEOUT) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return OpenAIExplanation(
            text=None,
            model=OPENAI_EXPLANATION_MODEL,
            latency_ms=latency_ms,
            error=f"OpenAI API вернул ошибку {exc.response.status_code}.",
        )
    except Exception:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return OpenAIExplanation(
            text=None,
            model=OPENAI_EXPLANATION_MODEL,
            latency_ms=latency_ms,
            error="Не удалось получить ответ от OpenAI API.",
        )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    text = _extract_text(data)
    if not text:
        return OpenAIExplanation(
            text=None,
            model=OPENAI_EXPLANATION_MODEL,
            latency_ms=latency_ms,
            error="OpenAI ответил без текстового обоснования.",
        )

    return OpenAIExplanation(
        text=text,
        model=str(data.get("model") or OPENAI_EXPLANATION_MODEL),
        latency_ms=latency_ms,
        error=None,
    )
