from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


CLICKBAIT_WORDS = {
    "en": [
        "shocking",
        "sensational",
        "exclusive",
        "must see",
        "you won't believe",
        "breaking",
        "urgent",
        "secret",
        "miracle",
    ],
    "ru": [
        "шок",
        "сенсация",
        "срочно",
        "эксклюзив",
        "не поверите",
        "тайна",
        "бомба",
    ],
    "kk": [
        "шок",
        "сенсация",
        "шұғыл",
        "эксклюзив",
        "сенбейсіз",
        "құпия",
    ],
}

HEDGE_WORDS = {
    "en": ["allegedly", "reportedly", "rumor", "maybe", "unconfirmed"],
    "ru": ["якобы", "возможно", "по слухам", "неподтверждено", "предположительно"],
    "kk": ["мүмкін", "сыбыс", "болжам", "расталмаған"],
}

SOURCE_MARKERS = {
    "en": ["according to", "reuters", "associated press", "official statement", "ministry"],
    "ru": ["по данным", "сообщил", "заявил", "источник", "официально"],
    "kk": ["мәліметінше", "хабарлады", "ресми", "дереккөз"],
}

SOURCE_ABSENCE_MARKERS = {
    "en": [
        "no official publications",
        "no official sources",
        "no verified sources",
        "sources are not provided",
        "unverified",
        "not confirmed",
    ],
    "ru": [
        "официальных публикаций не",
        "официальных публикаций",
        "проверяемых источников не",
        "проверяемых источников",
        "официальных источников нет",
        "источников нет",
        "источников не представлено",
        "источники не указаны",
        "не представлено",
        "не представлены",
        "не указано",
        "не подтверждено",
        "без источников",
    ],
    "kk": [
        "ресми жарияланымдар жоқ",
        "дереккөздер көрсетілмеген",
        "расталмаған",
    ],
}

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі0-9]+")
ALL_CAPS_RE = re.compile(r"\b[A-ZА-ЯЁ]{4,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


@dataclass
class XAIReason:
    reason_type: str
    text: str
    weight: float
    code: str


@dataclass
class XAIReport:
    reasons: List[XAIReason]
    highlights: List[Dict[str, str]]
    compact_codes: List[str]
    detailed_summary: str
    decision_path: List[str]
    feature_snapshot: Dict[str, float]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _keyword_hits(text: str, words: Sequence[str]) -> List[str]:
    normalized = _norm(text)
    hits: List[str] = []
    for w in words:
        needle = _norm(w)
        if not needle:
            continue
        # Phrases are matched as substrings, single-token markers require word boundaries.
        if " " in needle or "-" in needle:
            if needle in normalized:
                hits.append(w)
        else:
            if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", normalized):
                hits.append(w)
    return sorted(set(hits))


def _headline_candidate(text: str) -> str:
    line = text.split("\n", 1)[0].strip()
    if 20 <= len(line) <= 180:
        return line
    # fallback: first sentence
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence[:180]


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _fmt_pct(v: float) -> str:
    return f"{v * 100.0:.1f}%"


def extract_meta_features(text: str, language: str) -> Dict[str, float]:
    txt = text or ""
    words = WORD_RE.findall(txt)
    word_count = len(words)
    char_count = len(txt)
    upper_count = sum(1 for ch in txt if ch.isupper())
    letter_count = sum(1 for ch in txt if ch.isalpha())
    exclam = txt.count("!")
    question = txt.count("?")
    digits = sum(1 for ch in txt if ch.isdigit())
    urls = len(URL_RE.findall(txt))
    all_caps = len(ALL_CAPS_RE.findall(txt))

    vocab = Counter(w.lower() for w in words)
    unique_ratio = (len(vocab) / word_count) if word_count else 0.0

    clickbait_hits = len(_keyword_hits(txt, CLICKBAIT_WORDS.get(language, [])))
    hedge_hits = len(_keyword_hits(txt, HEDGE_WORDS.get(language, [])))
    source_hits = len(_keyword_hits(txt, SOURCE_MARKERS.get(language, [])))
    source_absence_hits = len(_keyword_hits(txt, SOURCE_ABSENCE_MARKERS.get(language, [])))

    return {
        "word_count": float(word_count),
        "char_count": float(char_count),
        "avg_word_len": float(sum(len(w) for w in words) / word_count) if word_count else 0.0,
        "uppercase_ratio": float(upper_count / max(letter_count, 1)),
        "exclamation_density": float(exclam / max(word_count, 1)),
        "question_density": float(question / max(word_count, 1)),
        "digit_density": float(digits / max(char_count, 1)),
        "url_count": float(urls),
        "all_caps_words": float(all_caps),
        "unique_ratio": float(unique_ratio),
        "clickbait_hits": float(clickbait_hits),
        "hedge_hits": float(hedge_hits),
        "source_hits": float(source_hits),
        "source_absence_hits": float(source_absence_hits),
    }


def build_xai_report(
    *,
    text_original: str,
    detected_language: str,
    fake_probability: float,
    rf_probability: float | None = None,
) -> XAIReport:
    lang = detected_language if detected_language in CLICKBAIT_WORDS else "en"
    features = extract_meta_features(text_original, lang)
    reasons: List[XAIReason] = []
    highlights: List[Dict[str, str]] = []
    codes: List[str] = []

    clickbait_terms = _keyword_hits(text_original, CLICKBAIT_WORDS[lang])
    if clickbait_terms:
        score = _clip(0.45 + 0.08 * len(clickbait_terms))
        reasons.append(
            XAIReason(
                reason_type="explicit",
                text=f"Частые кликбейт-слова: {', '.join(clickbait_terms[:5])}.",
                weight=score,
                code="KW_CLICKBAIT",
            )
        )
        codes.append("KW_CLICKBAIT")
        for term in clickbait_terms[:6]:
            highlights.append({"fragment": term, "reason": "кликбейт-лексика"})

    headline = _headline_candidate(text_original)
    if headline:
        headline_flags = 0
        if headline.count("!") >= 2:
            headline_flags += 1
        if headline.count("?") >= 1:
            headline_flags += 1
        if re.search(r"\b\d+\b", headline):
            headline_flags += 1
        if len(headline) < 70 and headline.upper() == headline and len(headline) > 20:
            headline_flags += 1
        if headline_flags >= 2:
            weight = _clip(0.4 + 0.12 * headline_flags)
            reasons.append(
                XAIReason(
                    reason_type="explicit",
                    text="Структура заголовка соответствует паттерну кликбейта.",
                    weight=weight,
                    code="HEADLINE_CLICKBAIT",
                )
            )
            codes.append("HEADLINE_CLICKBAIT")

    hedge_terms = _keyword_hits(text_original, HEDGE_WORDS[lang])
    source_terms = _keyword_hits(text_original, SOURCE_MARKERS[lang])
    source_absence_terms = _keyword_hits(text_original, SOURCE_ABSENCE_MARKERS[lang])
    if hedge_terms:
        reasons.append(
            XAIReason(
                reason_type="implicit",
                text=f"Обнаружены маркеры неопределенности: {', '.join(hedge_terms[:4])}.",
                weight=_clip(0.3 + 0.06 * len(hedge_terms)),
                code="HEDGE_WORDS",
            )
        )
        codes.append("HEDGE_WORDS")

    if source_absence_terms:
        reasons.append(
            XAIReason(
                reason_type="implicit",
                text=f"Найдены формулировки об отсутствии подтверждённых источников: {', '.join(source_absence_terms[:4])}.",
                weight=_clip(0.40 + 0.07 * len(source_absence_terms)),
                code="LOW_SOURCE_EVIDENCE",
            )
        )
        codes.append("LOW_SOURCE_EVIDENCE")
        for term in source_absence_terms[:4]:
            highlights.append({"fragment": term, "reason": "нет подтверждённых источников"})

    if source_terms and not source_absence_terms:
        reasons.append(
            XAIReason(
                reason_type="context",
                text=f"В тексте присутствуют маркеры проверяемых источников: {', '.join(source_terms[:4])}.",
                weight=_clip(0.28 + 0.05 * len(source_terms)),
                code="SOURCE_EVIDENCE_PRESENT",
            )
        )
        codes.append("SOURCE_EVIDENCE_PRESENT")
        for term in source_terms[:5]:
            highlights.append({"fragment": term, "reason": "маркер источника"})

    if features["source_hits"] == 0 and features["source_absence_hits"] == 0 and features["word_count"] >= 40:
        reasons.append(
            XAIReason(
                reason_type="implicit",
                text="В тексте почти нет ссылок на проверяемые источники.",
                weight=0.42,
                code="LOW_SOURCE_EVIDENCE",
            )
        )
        codes.append("LOW_SOURCE_EVIDENCE")

    if features["uppercase_ratio"] > 0.16 or features["all_caps_words"] >= 3:
        reasons.append(
            XAIReason(
                reason_type="implicit",
                text="Повышенная доля CAPS/эмоционального оформления.",
                weight=_clip(0.32 + features["uppercase_ratio"]),
                code="STYLE_EMOTIONAL",
            )
        )
        codes.append("STYLE_EMOTIONAL")
        for caps in ALL_CAPS_RE.findall(text_original)[:5]:
            highlights.append({"fragment": caps, "reason": "эмоциональный CAPS"})

    if features["exclamation_density"] > 0.02:
        reasons.append(
            XAIReason(
                reason_type="implicit",
                text="Слишком много восклицательных маркеров для нейтральной новости.",
                weight=_clip(0.2 + features["exclamation_density"] * 5),
                code="PUNC_EXCESS",
            )
        )
        codes.append("PUNC_EXCESS")

    if not clickbait_terms and features["word_count"] >= 35:
        reasons.append(
            XAIReason(
                reason_type="context",
                text="Не найдено кликбейт-лексики; лексика ближе к нейтральному новостному стилю.",
                weight=0.26,
                code="NEUTRAL_VOCAB_PATTERN",
            )
        )
        codes.append("NEUTRAL_VOCAB_PATTERN")

    if features["exclamation_density"] <= 0.01 and features["uppercase_ratio"] < 0.08:
        reasons.append(
            XAIReason(
                reason_type="context",
                text="Стиль текста сдержанный: нет избытка CAPS и эмоциональной пунктуации.",
                weight=0.24,
                code="CALM_STYLE",
            )
        )
        codes.append("CALM_STYLE")

    conf_margin = abs(fake_probability - (1.0 - fake_probability))
    if fake_probability >= 0.75:
        reasons.append(
            XAIReason(
                reason_type="context",
                text=f"Модель уверенно относит текст к fake (margin={conf_margin:.2f}).",
                weight=_clip(0.35 + conf_margin * 0.6),
                code="MODEL_HIGH_FAKE_CONF",
            )
        )
        codes.append("MODEL_HIGH_FAKE_CONF")
    elif fake_probability <= 0.25:
        reasons.append(
            XAIReason(
                reason_type="context",
                text=f"Модель даёт низкий риск фейка ({_fmt_pct(fake_probability)}), что поддерживает класс «достоверно».",
                weight=_clip(0.26 + (0.25 - fake_probability) * 0.8),
                code="MODEL_LOW_FAKE_CONF",
            )
        )
        codes.append("MODEL_LOW_FAKE_CONF")

    if rf_probability is not None:
        if rf_probability >= 0.70:
            reasons.append(
                XAIReason(
                    reason_type="context",
                    text="Random Forest подтверждает риск фейка по структурным признакам.",
                    weight=_clip(0.3 + (rf_probability - 0.5)),
                    code="RF_SUPPORT_FAKE",
                )
            )
            codes.append("RF_SUPPORT_FAKE")
        elif rf_probability <= 0.30 and fake_probability >= 0.60:
            reasons.append(
                XAIReason(
                    reason_type="context",
                    text="Есть расхождение моделей: вероятна неоднозначность источника.",
                    weight=0.25,
                    code="MODEL_DISAGREEMENT",
                )
            )
            codes.append("MODEL_DISAGREEMENT")

    if not reasons:
        reasons.append(
            XAIReason(
                reason_type="context",
                text="Явных триггеров не найдено, решение основано на суммарном языковом паттерне.",
                weight=0.2,
                code="NO_STRONG_RULES",
            )
        )
        codes.append("NO_STRONG_RULES")

    # Stable ordering keeps payload short and deterministic.
    reasons.sort(key=lambda x: x.weight, reverse=True)
    uniq_codes = []
    for c in codes:
        if c not in uniq_codes:
            uniq_codes.append(c)

    dedup_highlights = []
    seen_fragments = set()
    for h in highlights:
        key = h["fragment"].strip().lower()
        if key and key not in seen_fragments:
            dedup_highlights.append(h)
            seen_fragments.add(key)

    feature_snapshot = {
        "word_count": round(features["word_count"], 0),
        "clickbait_hits": round(features["clickbait_hits"], 0),
        "hedge_hits": round(features["hedge_hits"], 0),
        "source_hits": round(features["source_hits"], 0),
        "source_absence_hits": round(features["source_absence_hits"], 0),
        "uppercase_ratio": round(features["uppercase_ratio"], 3),
        "exclamation_density": round(features["exclamation_density"], 3),
        "url_count": round(features["url_count"], 0),
    }

    decision_path = [
        f"Шаг 1. Язык: {lang.upper()}, длина текста: {int(features['word_count'])} слов.",
        f"Шаг 2. Явные маркеры: кликбейт={int(features['clickbait_hits'])}, неопределенность={int(features['hedge_hits'])}.",
        (
            f"Шаг 3. Источники: source-маркеры={int(features['source_hits'])}, "
            f"отсутствие-источников={int(features['source_absence_hits'])}, URL={int(features['url_count'])}."
        ),
        (
            f"Шаг 4. Стиль: CAPS ratio={features['uppercase_ratio']:.3f}, "
            f"восклицательность={features['exclamation_density']:.3f}."
        ),
        f"Шаг 5. Итоговая вероятность фейка: {_fmt_pct(fake_probability)}.",
    ]
    if rf_probability is not None:
        decision_path.append(f"Шаг 6. Проверка RF-метамоделью: fake={_fmt_pct(rf_probability)}.")

    if fake_probability >= 0.6:
        detailed_summary = (
            "Решение склоняется к «фейк»: обнаружены сигналы манипулятивной подачи/стиля, "
            "и модель фиксирует повышенный риск недостоверности. Наибольший вклад дали причины вверху списка."
        )
    elif fake_probability <= 0.4:
        detailed_summary = (
            "Решение склоняется к «достоверно»: текст выглядит нейтральным, есть признаки ссылочности "
            "и отсутствуют выраженные триггеры кликбейта/эмоционального давления. "
            "Итог подтверждён низкой вероятностью фейка."
        )
    else:
        detailed_summary = (
            "Решение пограничное: часть признаков указывает на нейтральность, часть — на риск манипуляции. "
            "Такой кейс стоит дополнительно проверить по первоисточникам."
        )

    return XAIReport(
        reasons=reasons[:6],
        highlights=dedup_highlights[:8],
        compact_codes=uniq_codes[:8],
        detailed_summary=detailed_summary,
        decision_path=decision_path,
        feature_snapshot=feature_snapshot,
    )


def short_signal_prompt() -> str:
    # Minimal-token instruction for agentic post-processing.
    return (
        "FAKE if >=2: KW_CLICKBAIT, HEADLINE_CLICKBAIT, LOW_SOURCE_EVIDENCE, "
        "STYLE_EMOTIONAL, PUNC_EXCESS, MODEL_HIGH_FAKE_CONF. "
        "REAL if source markers high and fake_prob<0.45."
    )


def short_signal_prompt_lang(language: str) -> str:
    lang = (language or "").lower()
    if lang == "ru":
        return (
            "ФЕЙК, если >=2 сигнала: KW_CLICKBAIT, HEADLINE_CLICKBAIT, "
            "LOW_SOURCE_EVIDENCE, STYLE_EMOTIONAL, PUNC_EXCESS, MODEL_HIGH_FAKE_CONF. "
            "ДОСТОВЕРНО, если source markers высокие и fake_prob<0.45."
        )
    return short_signal_prompt()
