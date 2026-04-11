from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Literal


ValidationCode = Literal["too_short", "gibberish"]

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі0-9]+")

LATIN_VOWELS = set("aeiouy")
CYRILLIC_VOWELS = set("аеёиоуыэюяәіөүұү")

STOPWORDS = {
    "latin": {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "have",
        "will",
        "into",
        "about",
        "after",
        "before",
        "official",
        "report",
    },
    "cyrillic": {
        "и",
        "в",
        "во",
        "на",
        "по",
        "с",
        "со",
        "к",
        "из",
        "от",
        "для",
        "что",
        "это",
        "как",
        "не",
        "о",
        "об",
        "при",
        "после",
        "ресми",
        "және",
        "үшін",
        "деп",
        "бар",
        "бір",
    },
}

MIN_MEANINGFUL_WORDS = 20


@dataclass
class TextValidationResult:
    is_valid: bool
    code: ValidationCode | None
    message: str | None
    cleaned_text: str
    word_count: int
    char_count: int
    diagnostics: Dict[str, float]

    def to_detail(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "diagnostics": self.diagnostics,
        }


class TextValidationError(ValueError):
    def __init__(self, result: TextValidationResult):
        super().__init__(result.message or "Input validation failed.")
        self.result = result


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _script_kind(words: Iterable[str]) -> str:
    latin = 0
    cyrillic = 0
    for word in words:
        for ch in word.lower():
            if "a" <= ch <= "z":
                latin += 1
            elif "а" <= ch <= "я" or ch in "ёәғқңөұүһі":
                cyrillic += 1
    return "latin" if latin >= cyrillic else "cyrillic"


def _has_vowel(word: str, script_kind: str) -> bool:
    vowels = LATIN_VOWELS if script_kind == "latin" else CYRILLIC_VOWELS
    return any(ch in vowels for ch in word.lower())


def _max_consonant_run(word: str, script_kind: str) -> int:
    vowels = LATIN_VOWELS if script_kind == "latin" else CYRILLIC_VOWELS
    best = 0
    current = 0
    for ch in word.lower():
        if not ch.isalpha():
            current = 0
            continue
        if ch in vowels:
            current = 0
            continue
        current += 1
        best = max(best, current)
    return best


def validate_news_text(text: str) -> TextValidationResult:
    cleaned = _clean_text(text)
    words = [w for w in WORD_RE.findall(cleaned) if any(ch.isalpha() for ch in w)]
    word_count = len(words)
    char_count = len(cleaned)

    if word_count < MIN_MEANINGFUL_WORDS:
        return TextValidationResult(
            is_valid=False,
            code="too_short",
            message="Слишком мало информации для анализа. Добавьте хотя бы 20 слов связного текста.",
            cleaned_text=cleaned,
            word_count=word_count,
            char_count=char_count,
            diagnostics={"min_required_words": float(MIN_MEANINGFUL_WORDS)},
        )

    script_kind = _script_kind(words)
    vowel_words = sum(1 for word in words if _has_vowel(word, script_kind))
    stopword_hits = sum(1 for word in words if word.lower() in STOPWORDS[script_kind])
    avg_word_len = sum(len(word) for word in words) / max(len(words), 1)
    consonant_heavy_words = sum(1 for word in words if _max_consonant_run(word, script_kind) >= 5)
    long_words = sum(1 for word in words if len(word) >= 14)

    vowel_ratio = vowel_words / max(len(words), 1)
    consonant_heavy_ratio = consonant_heavy_words / max(len(words), 1)
    long_word_ratio = long_words / max(len(words), 1)

    diagnostics = {
        "vowel_ratio": round(vowel_ratio, 3),
        "stopword_hits": float(stopword_hits),
        "avg_word_len": round(avg_word_len, 3),
        "consonant_heavy_ratio": round(consonant_heavy_ratio, 3),
        "long_word_ratio": round(long_word_ratio, 3),
    }

    gibberish_like = (
        (vowel_ratio < 0.45 and consonant_heavy_ratio > 0.35)
        or (avg_word_len > 11.5 and stopword_hits == 0 and consonant_heavy_ratio > 0.22)
        or (long_word_ratio > 0.55 and stopword_hits == 0 and vowel_ratio < 0.58)
        or (stopword_hits == 0 and consonant_heavy_ratio > 0.60 and avg_word_len > 7.0)
    )

    if gibberish_like:
        return TextValidationResult(
            is_valid=False,
            code="gibberish",
            message="Текст выглядит как случайный набор символов или слов без связного смысла. Вставьте нормальный фрагмент новости.",
            cleaned_text=cleaned,
            word_count=word_count,
            char_count=char_count,
            diagnostics=diagnostics,
        )

    return TextValidationResult(
        is_valid=True,
        code=None,
        message=None,
        cleaned_text=cleaned,
        word_count=word_count,
        char_count=char_count,
        diagnostics=diagnostics,
    )


def ensure_valid_news_text(text: str) -> TextValidationResult:
    result = validate_news_text(text)
    if not result.is_valid:
        raise TextValidationError(result)
    return result
