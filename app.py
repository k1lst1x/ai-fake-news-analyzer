from __future__ import annotations

import html
import json
import os
import re
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import requests
import streamlit as st

from backend.analytics_store import AnalyticsStore
from backend.config import ANALYTICS_DB_PATH


st.set_page_config(
    page_title="Анализатор фейк-новостей",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Merriweather:wght@400;700&display=swap');

:root {
  --bg-main: #fdf7ee;
  --bg-glow-1: #ffd8ab66;
  --bg-glow-2: #f6b57859;
  --bg-card: #fff8ed;
  --bg-card-strong: #fff2dd;
  --line: #e9cfaa;
  --line-strong: #e1b780;
  --ink: #3f2618;
  --muted: #8a6a4d;
  --accent: #c9642c;
  --accent-2: #f2a85a;
  --ok: #2e7b66;
  --warn: #b34832;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 420px at 8% -12%, var(--bg-glow-1), transparent 72%),
    radial-gradient(860px 500px at 110% 0%, var(--bg-glow-2), transparent 76%),
    linear-gradient(180deg, var(--bg-main), #fffdf8 42%, #fff8ef);
  color: var(--ink);
  font-family: 'Manrope', sans-serif;
}

[data-testid="stAppViewBlockContainer"] {
  max-width: 1320px;
  padding-top: 1.15rem;
  padding-bottom: 2rem;
}

[data-testid="stHorizontalBlock"] {
  gap: 0.9rem;
}

h1, h2, h3 {
  color: var(--ink);
  font-family: 'Merriweather', serif;
  letter-spacing: -0.01em;
}

h1 {
  margin-top: 0.45rem;
  margin-bottom: 0.45rem;
}

h2 {
  margin-top: 1.2rem;
  margin-bottom: 0.5rem;
}

h3 {
  margin-top: 0.85rem;
  margin-bottom: 0.45rem;
}

.sub {
  color: var(--muted);
  max-width: 900px;
  line-height: 1.55;
  margin-top: 0.35rem;
  margin-bottom: 1.1rem;
}

p, li {
  line-height: 1.62;
}

ul, ol {
  padding-left: 1.2rem;
}

.card {
  background: linear-gradient(180deg, var(--bg-card), #fff5e4 94%);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 18px 20px;
  box-shadow: 0 12px 24px rgba(104, 60, 27, 0.08);
  line-height: 1.58;
}

.hero-card {
  background:
    radial-gradient(520px 220px at 90% -20%, #ffd8a680, transparent 70%),
    linear-gradient(132deg, #fff6e9, #ffefd9 58%, #ffe4c3);
  border: 1px solid #e6c398;
  border-radius: 24px;
  padding: 24px;
  box-shadow: 0 14px 28px rgba(99, 55, 24, 0.12);
}

.hero-title {
  margin: 0 0 10px 0;
  font-size: 2.1rem;
  line-height: 1.1;
  letter-spacing: -0.02em;
}

.hero-sub {
  margin: 0;
  max-width: 900px;
  color: #7c5a40;
  line-height: 1.58;
}

.metric-good { color: var(--ok); font-weight: 800; }
.metric-bad { color: var(--warn); font-weight: 800; }

.kpi-card {
  background: linear-gradient(130deg, #fff9ef 0%, #fff3e3 55%, #ffe8cf 100%);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  min-height: 118px;
  box-shadow: 0 10px 22px rgba(110, 66, 31, 0.08);
}

.kpi-title {
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.kpi-value {
  color: var(--ink);
  font-size: 2rem;
  line-height: 1.1;
  margin-top: 8px;
  font-weight: 800;
  font-family: 'Merriweather', serif;
}

.kpi-note {
  color: #9a7a58;
  margin-top: 7px;
  font-size: 0.84rem;
}

.reason-item {
  background: #fff3e0;
  border: 1px solid #efcca0;
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 8px;
  line-height: 1.56;
}

.result-banner {
  display: grid;
  grid-template-columns: 1.45fr 1fr;
  gap: 16px;
  align-items: stretch;
}

.result-card-main {
  background: linear-gradient(130deg, #fff9ef 0%, #ffeed9 70%, #ffe1c0 100%);
  border: 1px solid #e9c79e;
  border-radius: 18px;
  padding: 16px 18px;
}

.result-card-side {
  background: linear-gradient(180deg, #fff8eb, #fff0dc);
  border: 1px solid #e8cb9f;
  border-radius: 18px;
  padding: 16px 18px;
}

.verdict-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border-radius: 999px;
  padding: 6px 12px;
  font-weight: 700;
  font-size: 0.92rem;
  border: 1px solid transparent;
}

.verdict-fake {
  background: #ffe5dc;
  border-color: #efb7a7;
  color: #9f2f1d;
}

.verdict-real {
  background: #dff4ea;
  border-color: #a8d8c4;
  color: #1f6a53;
}

.confidence-label {
  margin-top: 8px;
  color: #83593b;
  font-weight: 600;
}

.meter-track {
  margin-top: 10px;
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #f3debe;
  overflow: hidden;
}

.meter-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #d5672c, #ea9b53);
}

.hl {
  background: #ffdba8;
  padding: 2px 4px;
  border-radius: 6px;
}

.pill {
  display: inline-block;
  border: 1px solid #e8c392;
  background: #fff0d7;
  border-radius: 999px;
  padding: 4px 11px;
  margin-right: 6px;
  margin-bottom: 8px;
  font-size: 12px;
  color: #7a4e2f;
  font-weight: 600;
}

.legend-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin: 0.25rem 0 0.6rem 0;
}

.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border-radius: 999px;
  background: #fff2df;
  border: 1px solid #e8c59a;
  padding: 5px 11px;
  font-size: 0.85rem;
  color: #6b452f;
  font-weight: 700;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
}

.legend-dot-real {
  background: #2f8a78;
}

.legend-dot-fake {
  background: #d04d2f;
  box-shadow: 0 0 0 2px #ffd9c8;
}

.stButton > button[kind="primary"],
[data-testid="stDownloadButton"] > button {
  background: linear-gradient(90deg, #e07d3e 0%, #f3ad65 100%);
  color: #2e1c12;
  border: 1px solid #d38d4f;
  border-radius: 14px;
  font-weight: 700;
  min-height: 44px;
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stDownloadButton"] > button:hover {
  border-color: #be7335;
  filter: saturate(1.06);
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(125, 67, 24, 0.17);
}

.stButton > button[kind="secondary"] {
  background: #fff3df;
  color: #5d3b27;
  border: 1px solid #e2b987;
  border-radius: 14px;
  font-weight: 700;
  min-height: 44px;
  transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
}

.stButton > button[kind="secondary"]:hover {
  background: #ffe9c8;
  border-color: #d8a266;
  transform: translateY(-1px);
  box-shadow: 0 8px 15px rgba(110, 64, 28, 0.10);
}

.nav-caption {
  color: #865f42;
  font-size: 0.86rem;
  margin: 0 0 0.25rem 0.1rem;
  font-weight: 700;
  letter-spacing: 0.03em;
}

.nav-spacer {
  height: 0.7rem;
}

[data-testid="stWidgetLabel"] p,
label {
  color: #734e35 !important;
  opacity: 1 !important;
  font-weight: 700 !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-baseweb="input"] input,
textarea,
textarea::placeholder,
[data-baseweb="input"] input::placeholder {
  color: #5b3b28 !important;
  opacity: 1 !important;
}

[data-baseweb="select"] svg {
  fill: #8a5a36 !important;
}

[data-testid="stDataFrame"] {
  background: transparent;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
  background: #fff6ea !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 12px !important;
  color: #4a2d1d !important;
  font-size: 1rem !important;
  line-height: 1.55 !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover,
textarea:hover {
  border-color: #cf9355 !important;
}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
textarea:focus {
  border-color: #c56f34 !important;
  box-shadow: 0 0 0 1px #c56f34 !important;
}

[data-testid="stVegaLiteChart"] > div {
  border-radius: 16px;
  border: 1px solid var(--line);
  overflow: hidden;
  background: #fffaf2;
  box-shadow: 0 10px 22px rgba(106, 64, 33, 0.09);
}

[data-testid="stForm"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 10px 14px 4px 14px;
  background: linear-gradient(180deg, #fffaf1, #fff4e3);
  box-shadow: 0 8px 18px rgba(100, 57, 24, 0.08);
}

.table-wrap {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  background: #fff9ef;
  box-shadow: 0 8px 18px rgba(100, 57, 24, 0.08);
}

.preview-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.93rem;
}

.preview-table thead th {
  background: #fcebd2;
  color: #5d3c28;
  text-align: left;
  font-weight: 800;
  padding: 10px 10px;
  border-bottom: 1px solid #e8c99e;
}

.preview-table tbody td {
  padding: 9px 10px;
  border-bottom: 1px solid #f2dfc2;
  color: #4f311f;
}

.preview-table tbody tr:nth-child(even) {
  background: #fff4e2;
}

.preview-table tbody tr:last-child td {
  border-bottom: none;
}

hr {
  border-color: #ecd7bc;
}

@media (max-width: 1024px) {
  .card {
    padding: 14px 16px;
    border-radius: 14px;
  }
  .kpi-value {
    font-size: 1.7rem;
  }
  .hero-title {
    font-size: 1.8rem;
  }
  .result-banner {
    grid-template-columns: 1fr;
  }
  h1 {
    font-size: 1.9rem;
  }
}

@media (max-width: 768px) {
  .card {
    padding: 12px 12px;
    border-radius: 12px;
  }
  .kpi-card {
    min-height: 104px;
    padding: 12px 12px;
  }
  .kpi-value {
    font-size: 1.45rem;
  }
  .hero-card {
    padding: 16px;
    border-radius: 16px;
  }
  .hero-title {
    font-size: 1.45rem;
  }
  h1 {
    font-size: 1.6rem;
  }
  .sub {
    font-size: 0.95rem;
  }
  .pill {
    font-size: 11px;
    padding: 3px 8px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


def _resolve_api_base() -> str:
    try:
        val = st.secrets["API_BASE_URL"]
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


API_BASE = _resolve_api_base()
PAGE_KEYS = ["Приветственная", "Проверка", "Аналитический дашборд"]
LANG_LABELS = {"auto": "Авто", "ru": "Русский", "en": "Английский", "kk": "Казахский"}
CONTENT_TYPE_LABELS = {
    "article": "Статья",
    "post": "Пост",
    "headline": "Заголовок",
    "other": "Другое",
}
CLASS_LABELS = {"fake": "Фейк", "real": "Достоверная"}
REASON_TYPE_LABELS = {
    "explicit": "Явный признак",
    "implicit": "Неявный признак",
    "rf_meta": "Статистический признак",
    "context": "Контекстный признак",
}


def _warm_altair_theme() -> Dict[str, Any]:
    return {
        "config": {
            "background": "#fffaf2",
            "view": {"stroke": "#e9cfaa", "strokeWidth": 1},
            "axis": {
                "labelColor": "#6d4a33",
                "titleColor": "#6d4a33",
                "gridColor": "#f2e3cc",
                "domainColor": "#e0c29b",
                "tickColor": "#dcb990",
                "labelFont": "Manrope",
                "titleFont": "Manrope",
            },
            "legend": {
                "labelColor": "#5c3d2b",
                "titleColor": "#5c3d2b",
                "labelFont": "Manrope",
                "titleFont": "Manrope",
            },
            "title": {"color": "#4a2d1d", "font": "Merriweather"},
            "range": {
                "category": [
                    "#c9642c",
                    "#2e7b66",
                    "#df9345",
                    "#6f8cba",
                    "#7d5a45",
                ]
            },
        }
    }


if "warm_fake_news" not in alt.themes.names():
    alt.themes.register("warm_fake_news", _warm_altair_theme)
alt.themes.enable("warm_fake_news")


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _kpi_card(title: str, value: str, note: str) -> str:
    return (
        "<div class='kpi-card'>"
        f"<div class='kpi-title'>{html.escape(title)}</div>"
        f"<div class='kpi-value'>{html.escape(value)}</div>"
        f"<div class='kpi-note'>{html.escape(note)}</div>"
        "</div>"
    )


def _confidence_label(fake_prob: float) -> str:
    confidence = abs(fake_prob - 0.5) * 2.0
    if confidence >= 0.7:
        return "Высокая уверенность"
    if confidence >= 0.4:
        return "Средняя уверенность"
    return "Пограничный случай"


def _filter_pills(period: int, langs: List[str], types: List[str]) -> str:
    parts = [f"<span class='pill'>Период: {'Все время' if period == 0 else f'{period} дней'}</span>"]
    if langs:
        parts.append(f"<span class='pill'>Языки: {', '.join(langs)}</span>")
    else:
        parts.append("<span class='pill'>Языки: все</span>")
    if types:
        pretty_types = [CONTENT_TYPE_LABELS.get(t, t) for t in types]
        parts.append(f"<span class='pill'>Типы: {', '.join(pretty_types)}</span>")
    else:
        parts.append("<span class='pill'>Типы: все</span>")
    return "".join(parts)


def _render_top_nav() -> None:
    st.markdown("<p class='nav-caption'>Разделы проекта</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, page in enumerate(PAGE_KEYS):
        btn_type = "primary" if st.session_state.page == page else "secondary"
        if cols[idx].button(page, key=f"top_nav_{idx}", type=btn_type, use_container_width=True):
            st.session_state.page = page
            st.rerun()
    st.markdown("<div class='nav-spacer'></div>", unsafe_allow_html=True)


def _render_preview_table(preview: pd.DataFrame) -> None:
    pretty = preview.fillna("—").copy()
    table_html = pretty.to_html(index=False, escape=True, classes="preview-table")
    st.markdown(f"<div class='table-wrap'>{table_html}</div>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = PAGE_KEYS[0]
elif st.session_state.page == "Landing":
    st.session_state.page = "Приветственная"
elif st.session_state.page == "Главная":
    st.session_state.page = "Приветственная"
elif st.session_state.page == "Дашборд":
    st.session_state.page = "Аналитический дашборд"
elif st.session_state.page not in PAGE_KEYS:
    st.session_state.page = PAGE_KEYS[0]


@st.cache_resource
def get_local_service() -> Any:
    # Lazy import keeps UI fast and avoids heavy torch init on page load.
    from backend.service import NewsAnalyzerService

    return NewsAnalyzerService()


@st.cache_resource
def get_store() -> AnalyticsStore:
    return AnalyticsStore(ANALYTICS_DB_PATH)


@st.cache_data
def load_examples() -> Dict[str, str]:
    true_df = pd.read_csv("data/True.csv", nrows=1)
    fake_df = pd.read_csv("data/Fake.csv", nrows=1)
    return {
        "real": str(true_df.iloc[0]["text"])[:1400],
        "fake": str(fake_df.iloc[0]["text"])[:1400],
    }


def _local_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    service = get_local_service()
    store = get_store()
    res = service.analyze(text=payload["text"], language=payload.get("language", "auto"))

    if payload.get("store_event", True):
        store.log_check(
            language=res.detected_language,
            content_type=payload.get("content_type", "article"),
            country=payload.get("country"),
            label=res.label,
            fake_prob=res.fake_probability,
            real_prob=res.real_probability,
            text_length=len(payload["text"]),
            token_length=res.token_length,
            latency_ms=res.latency_ms,
            model_trace=res.model_trace,
        )

    return {
        "label": res.label,
        "fake_probability": res.fake_probability,
        "real_probability": res.real_probability,
        "detected_language": res.detected_language,
        "translated_to_english": res.translated_to_english,
        "translation_text": res.translation_text,
        "latency_ms": res.latency_ms,
        "explanation": res.explanation,
        "model_trace": res.model_trace,
    }


def analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(f"{API_BASE}/analyze", json=payload, timeout=6)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return _local_analyze(payload)


def highlight_text_html(text: str, highlights: List[Dict[str, str]]) -> str:
    if not highlights:
        return html.escape(text).replace("\n", "<br/>")

    result = text
    replacements: Dict[str, str] = {}
    ordered = []
    for h in highlights:
        frag = (h.get("fragment") or "").strip()
        if len(frag) < 3:
            continue
        if frag.lower() in {x.lower() for x in ordered}:
            continue
        ordered.append(frag)
    ordered.sort(key=len, reverse=True)

    for idx, frag in enumerate(ordered[:8]):
        token = f"@@H{idx}@@"
        pattern = re.compile(re.escape(frag), flags=re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(token, result, count=1)
            replacements[token] = frag

    escaped = html.escape(result)
    for token, frag in replacements.items():
        escaped = escaped.replace(
            html.escape(token),
            f"<mark class='hl' title='подозрительный фрагмент'>{html.escape(frag)}</mark>",
        )
    return escaped.replace("\n", "<br/>")


def render_prediction(result: Dict[str, Any], source_text: str) -> None:
    label = result["label"]
    fake_p = float(result["fake_probability"])
    real_p = float(result["real_probability"])
    latency_ms = int(result.get("latency_ms", 0))

    tone = "metric-bad" if label == "fake" else "metric-good"
    title = "Вероятно фейк" if label == "fake" else "Вероятно достоверно"
    badge_class = "verdict-fake" if label == "fake" else "verdict-real"
    confidence_text = _confidence_label(fake_p)
    fake_width = max(3.0, min(100.0, fake_p * 100.0))

    st.markdown(
        f"""
<div class="result-banner">
  <div class="result-card-main">
    <span class="verdict-badge {badge_class}">{title}</span>
    <div class="confidence-label">{confidence_text}</div>
    <div class="meter-track"><div class="meter-fill" style="width:{fake_width:.1f}%"></div></div>
    <p style="margin-top:10px;"><strong>Вероятность фейка:</strong> {fake_p * 100:.1f}%</p>
    <p><strong>Вероятность достоверности:</strong> {real_p * 100:.1f}%</p>
  </div>
  <div class="result-card-side">
    <h4 class="{tone}" style="margin:0 0 8px 0;">Технические параметры</h4>
    <p style="margin:0 0 6px 0;"><strong>Язык:</strong> {result.get("detected_language")}</p>
    <p style="margin:0 0 6px 0;"><strong>Задержка:</strong> {latency_ms} мс</p>
    <p style="margin:0;color:#8f6c4f;">Чем ниже задержка и выше уверенность, тем стабильнее ответ.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    explanation = result.get("explanation", {})
    reasons = explanation.get("reasons", [])
    detailed_summary = explanation.get("detailed_summary")
    decision_path = explanation.get("decision_path") or []
    feature_snapshot = explanation.get("feature_snapshot") or {}

    if detailed_summary:
        st.markdown("### Подробное обоснование")
        st.markdown(
            f"<div class='card'>{html.escape(str(detailed_summary))}</div>",
            unsafe_allow_html=True,
        )

    if decision_path:
        st.markdown("### Путь решения модели")
        st.markdown(
            "<div class='card'><ol style='margin:0; padding-left:20px;'>"
            + "".join(f"<li>{html.escape(str(step))}</li>" for step in decision_path[:8])
            + "</ol></div>",
            unsafe_allow_html=True,
        )

    if feature_snapshot:
        labels = {
            "word_count": "Слов",
            "clickbait_hits": "Кликбейт-маркеры",
            "hedge_hits": "Маркеры неопределенности",
            "source_hits": "Маркеры источников",
            "source_absence_hits": "Отсутствие источников",
            "uppercase_ratio": "Доля CAPS",
            "exclamation_density": "Плотность !",
            "url_count": "URL",
        }
        chips = []
        for k in [
            "word_count",
            "clickbait_hits",
            "hedge_hits",
            "source_hits",
            "source_absence_hits",
            "uppercase_ratio",
            "exclamation_density",
            "url_count",
        ]:
            if k in feature_snapshot:
                chips.append(f"<span class='pill'>{labels.get(k, k)}: {feature_snapshot[k]}</span>")
        if chips:
            st.markdown("**Диагностические признаки:**", unsafe_allow_html=True)
            st.markdown("".join(chips), unsafe_allow_html=True)

    st.markdown("### Почему модель так решила")
    for item in reasons:
        text = html.escape(item.get("text", ""))
        reason_type = item.get("type", "")
        reason_type_label = REASON_TYPE_LABELS.get(str(reason_type).lower(), str(reason_type))
        weight = float(item.get("weight", 0.0))
        code = item.get("code", "")
        st.markdown(
            f"""
<div class="reason-item">
  <strong>{reason_type_label}</strong> [{code}] · вес {weight:.2f}<br/>{text}
</div>
""",
            unsafe_allow_html=True,
        )

    compact_codes = explanation.get("compact_codes", [])
    if compact_codes:
        st.markdown("**Компактные сигналы (для агента, минимум токенов):**")
        st.markdown(
            "".join(f"<span class='pill'>{html.escape(code)}</span>" for code in compact_codes),
            unsafe_allow_html=True,
        )
        hint = explanation.get("agent_hint_min_tokens")
        if hint:
            st.code(hint, language="text")

    highlights = explanation.get("highlights", [])
    if highlights:
        st.markdown("### Подсветка подозрительных фрагментов")
        st.markdown(
            f"<div class='card' style='line-height:1.65;'>{highlight_text_html(source_text, highlights)}</div>",
            unsafe_allow_html=True,
        )

    if result.get("translated_to_english") and result.get("translation_text"):
        st.markdown("### Перевод, использованный моделью")
        st.markdown(
            f"<div class='card'><pre style='white-space:pre-wrap; margin:0;'>{html.escape(result['translation_text'])}</pre></div>",
            unsafe_allow_html=True,
        )

    st.caption(f"Трассировка модели: {result.get('model_trace', '-')}")


def landing_page() -> None:
    st.caption("Приветственная страница (Landing)")
    st.markdown(
        """
<div class="hero-card">
  <h1 class="hero-title">AI Fake News Analyzer</h1>
  <p class="hero-sub">
    Платформа для проверки новостей с Explainable AI: модель оценивает риск фейка, объясняет решение
    по признакам и сохраняет аналитику для исследований и отчетности.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    cta1, cta2, _ = st.columns([1.1, 1.1, 2.8])
    with cta1:
        if st.button("Перейти к проверке", type="primary", use_container_width=True):
            st.session_state.page = "Проверка"
            st.rerun()
    with cta2:
        if st.button("Открыть дашборд", use_container_width=True):
            st.session_state.page = "Аналитический дашборд"
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
<div class="card">
  <h3>Цели проекта</h3>
  <ul>
    <li>Выявлять фейковые новости на английском, русском и казахском.</li>
    <li>Показывать объяснение решения: явные и неявные признаки.</li>
    <li>Собирать статистику и экспорт для научного анализа.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="card">
  <h3>Что нового в версии диплома</h3>
  <ul>
    <li>Полноценный API + совместная проверка моделью и правилами.</li>
    <li>XAI-подсветка подозрительных фрагментов.</li>
    <li>Аналитический дашборд с экспортом CSV/JSON.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("## Примеры запросов")
    examples = load_examples()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Пример 1: Реальная новость")
        sample_real = examples["real"][:850]
        st.markdown(f"<div class='card'>{html.escape(sample_real)}...</div>", unsafe_allow_html=True)
        if st.button("Проверить пример: достоверная", use_container_width=True):
            result = analyze(
                {
                    "text": sample_real,
                    "language": "en",
                    "content_type": "article",
                    "country": "US",
                    "store_event": False,
                }
            )
            render_prediction(result, sample_real)

    with col_b:
        st.markdown("### Пример 2: Фейковая новость")
        sample_fake = examples["fake"][:850]
        st.markdown(f"<div class='card'>{html.escape(sample_fake)}...</div>", unsafe_allow_html=True)
        if st.button("Проверить пример: фейковая", use_container_width=True):
            result = analyze(
                {
                    "text": sample_fake,
                    "language": "en",
                    "content_type": "article",
                    "country": "US",
                    "store_event": False,
                }
            )
            render_prediction(result, sample_fake)

    if st.button("Попробовать", type="primary", use_container_width=True):
        st.session_state.page = "Проверка"
        st.rerun()


def analyzer_page() -> None:
    st.markdown("<h1>Проверка новости</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>Вставьте текст новости. Система вернёт вероятность «фейк/достоверная», уровень уверенности и подробное объяснение с подсветкой.</p>",
        unsafe_allow_html=True,
    )

    if "analyze_text" not in st.session_state:
        st.session_state.analyze_text = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_source_text" not in st.session_state:
        st.session_state.last_source_text = ""

    examples = load_examples()
    qb1, qb2, qb3 = st.columns(3)
    with qb1:
        if st.button("Вставить real-пример", use_container_width=True):
            st.session_state.analyze_text = examples["real"][:1400]
            st.rerun()
    with qb2:
        if st.button("Вставить fake-пример", use_container_width=True):
            st.session_state.analyze_text = examples["fake"][:1400]
            st.rerun()
    with qb3:
        if st.button("Очистить текст", use_container_width=True):
            st.session_state.analyze_text = ""
            st.session_state.last_result = None
            st.session_state.last_source_text = ""
            st.rerun()

    with st.expander("Как получить более точный результат"):
        st.markdown(
            "- Вставляйте полный абзац или статью, а не один заголовок.\n"
            "- Если известна страна/контекст, укажите ее в поле `Страна`.\n"
            "- Для спорных кейсов проверяйте первоисточники, даже если модель уверена."
        )

    with st.form("analyze_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
        with col1:
            language = st.selectbox(
                "Язык",
                ["auto", "ru", "en", "kk"],
                index=0,
                format_func=lambda code: LANG_LABELS.get(code, code),
            )
        with col2:
            content_type = st.selectbox(
                "Тип контента",
                ["article", "post", "headline", "other"],
                index=0,
                format_func=lambda kind: CONTENT_TYPE_LABELS.get(kind, kind),
            )
        with col3:
            country = st.text_input("Страна (опц.)", placeholder="RU, KZ, US ...")

        text = st.text_area(
            "Текст новости",
            height=260,
            placeholder="Вставьте новость для анализа...",
            key="analyze_text",
        )
        clean = text.strip()
        word_count = len(clean.split()) if clean else 0
        char_count = len(clean)
        st.caption(f"Объем текста: {word_count} слов · {char_count} символов")
        submitted = st.form_submit_button("Анализировать", type="primary", use_container_width=True)

    if submitted:
        if len(clean) < 10:
            st.warning("Нужен текст не короче 10 символов.")
            return
        payload = {
            "text": clean,
            "language": language,
            "content_type": content_type,
            "country": country.strip() or None,
            "store_event": True,
        }
        with st.spinner("Проверка и объяснение..."):
            result = analyze(payload)
        st.session_state.last_result = result
        st.session_state.last_source_text = clean

    if st.session_state.last_result:
        st.markdown("### Результат анализа")
        render_prediction(st.session_state.last_result, st.session_state.last_source_text)


def dashboard_page() -> None:
    st.markdown("<h1>Аналитический дашборд</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub'>Статистика проверок, распределения по времени, языкам и географии, плюс экспорт для исследований.</p>",
        unsafe_allow_html=True,
    )
    store = get_store()

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("Период", [7, 30, 90, 365, 0], format_func=lambda x: "Все время" if x == 0 else f"{x} дней")
    with c2:
        langs = st.multiselect("Языки", ["en", "ru", "kk"], default=[], placeholder="Все языки")
    with c3:
        types = st.multiselect(
            "Тип контента",
            ["article", "post", "headline", "other"],
            default=[],
            placeholder="Все типы",
            format_func=lambda kind: CONTENT_TYPE_LABELS.get(kind, kind),
        )

    st.markdown(_filter_pills(period, langs, types), unsafe_allow_html=True)
    st.markdown("<div class='nav-spacer'></div>", unsafe_allow_html=True)

    summary = store.summary(days=period, languages=langs, content_types=types)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            _kpi_card(
                title="Всего проверок",
                value=_fmt_int(summary.total_checks),
                note="Накопленный объем анализов",
            ),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            _kpi_card(
                title="Фейк",
                value=_fmt_int(summary.fake_checks),
                note="Классифицировано как fake",
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            _kpi_card(
                title="Достоверные",
                value=_fmt_int(summary.real_checks),
                note="Классифицировано как real",
            ),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            _kpi_card(
                title="Доля фейков",
                value=f"{summary.fake_share * 100:.1f}%",
                note=f"Средняя задержка: {summary.avg_latency_ms:.1f} мс",
            ),
            unsafe_allow_html=True,
        )

    if summary.total_checks > 0:
        fake_trend = "Нормально" if summary.fake_share < 0.35 else "Требует внимания"
        st.markdown(
            f"<div class='card'><strong>Состояние потока:</strong> {fake_trend}. "
            f"За период обработано {_fmt_int(summary.total_checks)} материалов.</div>",
            unsafe_allow_html=True,
        )

    timeline = store.timeline(days=period, languages=langs, content_types=types)
    st.markdown("### Динамика fake/real")
    if not timeline.empty:
        st.markdown(
            """
<div class="legend-row">
  <span class="legend-chip"><span class="legend-dot legend-dot-real"></span>Достоверная</span>
  <span class="legend-chip"><span class="legend-dot legend-dot-fake"></span>Фейк</span>
</div>
""",
            unsafe_allow_html=True,
        )
        timeline = timeline.copy()
        timeline["label_ru"] = timeline["label"].map(CLASS_LABELS).fillna(timeline["label"])
        timeline["bucket"] = timeline["bucket"].astype(str)
        labels_order = ["Достоверная", "Фейк"]
        idx = pd.MultiIndex.from_product(
            [sorted(timeline["bucket"].unique()), labels_order],
            names=["bucket", "label_ru"],
        )
        timeline = (
            timeline.set_index(["bucket", "label_ru"])["value"]
            .reindex(idx, fill_value=0)
            .reset_index()
        )
        timeline["is_fake"] = timeline["label_ru"].eq("Фейк")

        color_scale = alt.Scale(domain=labels_order, range=["#2f8a78", "#d04d2f"])
        base = alt.Chart(timeline).encode(
            x=alt.X(
                "bucket:N",
                sort=None,
                title="Дата",
                axis=alt.Axis(labelAngle=-20, labelLimit=120),
            ),
            y=alt.Y("value:Q", title="Проверки", scale=alt.Scale(zero=True)),
            color=alt.Color("label_ru:N", title="Класс", scale=color_scale),
            tooltip=["bucket:N", "label_ru:N", "value:Q"],
        )
        line_real = (
            base.transform_filter(alt.datum.label_ru == "Достоверная")
            .mark_line(strokeWidth=3.0, opacity=0.82, interpolate="monotone")
        )
        line_fake = (
            base.transform_filter(alt.datum.label_ru == "Фейк")
            .mark_line(strokeWidth=4.8, interpolate="monotone", strokeDash=[7, 4], opacity=1.0)
        )
        points_real = (
            base.transform_filter(alt.datum.label_ru == "Достоверная")
            .mark_point(size=86, filled=True, opacity=0.82)
        )
        points_fake = (
            base.transform_filter(alt.datum.label_ru == "Фейк")
            .mark_point(size=185, filled=True, shape="diamond", stroke="#fffaf2", strokeWidth=1.2)
        )
        st.altair_chart((line_real + line_fake + points_real + points_fake).properties(height=300), use_container_width=True)
    else:
        st.info("Пока нет данных для выбранных фильтров.")

    st.markdown("### Языки и география")
    lcol, gcol = st.columns(2)
    with lcol:
        lang_df = store.language_distribution(days=period, content_types=types)
        if not lang_df.empty:
            lang_chart = (
                alt.Chart(lang_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("bucket:N", title="Язык"),
                    y=alt.Y("value:Q", title="Проверки"),
                    color=alt.Color(
                        "bucket:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=["en", "ru", "kk"],
                            range=["#2e7b66", "#d07d42", "#6f8cba"],
                        ),
                    ),
                )
                .properties(height=220)
            )
            st.altair_chart(lang_chart, use_container_width=True)

    with gcol:
        geo_df = store.geo_distribution(days=period, languages=langs, content_types=types)
        if not geo_df.empty:
            geo_chart = (
                alt.Chart(geo_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("bucket:N", title="Страна"),
                    y=alt.Y("value:Q", title="Проверки"),
                    color=alt.Color(
                        "value:Q",
                        legend=None,
                        scale=alt.Scale(
                            range=["#fff2d8", "#f1b46f", "#d07a3f", "#b45c2a"],
                        ),
                    ),
                )
                .properties(height=220)
            )
            st.altair_chart(geo_chart, use_container_width=True)

    export_df = store.export_checks(days=period, languages=langs, content_types=types)
    if not export_df.empty:
        preview = export_df.copy().head(12)
        preview["label"] = preview["label"].map(CLASS_LABELS).fillna(preview["label"])
        preview = preview.rename(
            columns={
                "created_at": "Время",
                "language": "Язык",
                "content_type": "Тип",
                "country": "Страна",
                "label": "Класс",
                "fake_prob": "P(fake)",
                "latency_ms": "Задержка, мс",
            }
        )
        if "P(fake)" in preview.columns:
            preview["P(fake)"] = pd.to_numeric(preview["P(fake)"], errors="coerce").round(3)
        cols = ["Время", "Язык", "Тип", "Страна", "Класс", "P(fake)", "Задержка, мс"]
        st.markdown("### Последние проверки")
        _render_preview_table(preview[cols])

    st.markdown("### Экспорт данных")
    col_csv, col_json = st.columns(2)
    with col_csv:
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Скачать CSV",
            data=csv_bytes,
            file_name="fake_news_analytics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_json:
        json_bytes = json.dumps(export_df.to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "Скачать JSON",
            data=json_bytes,
            file_name="fake_news_analytics.json",
            mime="application/json",
            use_container_width=True,
        )


_render_top_nav()

if st.session_state.page == "Приветственная":
    landing_page()
elif st.session_state.page == "Проверка":
    analyzer_page()
else:
    dashboard_page()



