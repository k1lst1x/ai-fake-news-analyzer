from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.utils import logging as hf_logging

from .openai_explainer import generate_openai_explanation
from .config import (
    KK_EN_DIR,
    MAX_INPUT_CHARS,
    MAX_MODEL_TOKENS,
    MODEL_DIR,
    RF_MODEL_PATH,
    RU_EN_DIR,
)
from .text_validation import ensure_valid_news_text
from .xai import build_xai_report, extract_meta_features, short_signal_prompt_lang


CYRILLIC_RE = re.compile(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")
KAZAKH_ONLY_RE = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")

RULE_CFG = {
    "clickbait_base": 0.15,
    "clickbait_step": 0.04,
    "clickbait_cap": 0.24,
    "hedge_base": 0.10,
    "hedge_step": 0.03,
    "hedge_cap": 0.18,
    "absence_base": 0.20,
    "absence_step": 0.05,
    "absence_cap": 0.32,
    "no_source_bonus": 0.08,
    "style_bonus": 0.08,
    "punc_bonus": 0.06,
    "strong_fake_combo_bonus": 0.15,
    "real_source_discount": 0.10,
    "rf_real_discount": 0.04,
    "delta_min": -0.20,
    "delta_max": 0.70,
}

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


@dataclass
class AnalyzeResult:
    label: str
    fake_probability: float
    real_probability: float
    detected_language: str
    translated_to_english: bool
    translation_text: Optional[str]
    latency_ms: int
    explanation: Dict[str, Any]
    model_trace: str
    token_length: int


class NewsAnalyzerService:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clf_tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        self.clf_model.to(self.device).eval()
        self.fake_class_id = 0
        self.real_class_id = 1
        self._infer_label_mapping()

        self.ru_tok = None
        self.ru_model = None
        self.kk_tok = None
        self.kk_model = None

        self.rf_bundle = None
        if RF_MODEL_PATH.exists():
            self.rf_bundle = joblib.load(RF_MODEL_PATH)

    def _predict_raw_probs(self, text_en: str) -> tuple[list[float], int]:
        inputs = self.clf_tok(
            text_en,
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
            return_tensors="pt",
        )
        token_len = int(inputs["input_ids"].shape[1])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.clf_model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        return [float(x) for x in probs], token_len

    def _infer_label_mapping(self) -> None:
        # Some checkpoints keep generic labels (LABEL_0/LABEL_1) without semantics.
        # We infer mapping from two anchor probes.
        fake_probe = "Shocking exclusive secret reveal! You will not believe this claim!"
        real_probe = "Reuters reported an official ministry statement with documented data."
        try:
            fake_scores, _ = self._predict_raw_probs(fake_probe)
            real_scores, _ = self._predict_raw_probs(real_probe)
            if len(fake_scores) >= 2 and len(real_scores) >= 2:
                margin_0 = fake_scores[0] - real_scores[0]
                margin_1 = fake_scores[1] - real_scores[1]
                if margin_1 > margin_0:
                    self.fake_class_id = 1
                    self.real_class_id = 0
                else:
                    self.fake_class_id = 0
                    self.real_class_id = 1
        except Exception:
            # Fallback keeps backwards compatibility with older checkpoints.
            self.fake_class_id = 0
            self.real_class_id = 1

    @staticmethod
    def _folder_has_config(path: Path) -> bool:
        return path.is_dir() and (path / "config.json").exists()

    def _load_ru_translator(self) -> None:
        if self.ru_tok is not None and self.ru_model is not None:
            return
        if not self._folder_has_config(RU_EN_DIR):
            return
        self.ru_tok = AutoTokenizer.from_pretrained(RU_EN_DIR, local_files_only=True)
        self.ru_model = AutoModelForSeq2SeqLM.from_pretrained(RU_EN_DIR, local_files_only=True)
        self.ru_model.to(self.device).eval()

    def _load_kk_translator(self) -> None:
        if self.kk_tok is not None and self.kk_model is not None:
            return
        if not self._folder_has_config(KK_EN_DIR):
            return
        self.kk_tok = AutoTokenizer.from_pretrained(KK_EN_DIR, local_files_only=True)
        self.kk_model = AutoModelForSeq2SeqLM.from_pretrained(KK_EN_DIR, local_files_only=True)
        self.kk_model.to(self.device).eval()

    @staticmethod
    def _normalize_text(text: str) -> str:
        txt = (text or "").strip()
        txt = re.sub(r"\s+", " ", txt)
        # Controlled truncation keeps compute and token cost predictable.
        return txt[:MAX_INPUT_CHARS]

    @staticmethod
    def detect_language(text: str, hint: str = "auto") -> str:
        if hint in {"en", "ru", "kk"}:
            return hint
        if KAZAKH_ONLY_RE.search(text):
            return "kk"
        if CYRILLIC_RE.search(text):
            return "ru"
        return "en"

    def _translate(self, text: str, lang: str) -> tuple[str, bool, Optional[str]]:
        if lang == "en":
            return text, False, None
        if lang == "ru":
            self._load_ru_translator()
            if self.ru_tok is None or self.ru_model is None:
                return text, False, None
            tok = self.ru_tok
            mdl = self.ru_model
            if hasattr(tok, "lang_code_to_id") and hasattr(tok, "src_lang"):
                tok.src_lang = "rus_Cyrl"
            forced_bos = tok.lang_code_to_id.get("eng_Latn") if hasattr(tok, "lang_code_to_id") else None
        else:
            self._load_kk_translator()
            if self.kk_tok is None or self.kk_model is None:
                return text, False, None
            tok = self.kk_tok
            mdl = self.kk_model
            if hasattr(tok, "src_lang"):
                tok.src_lang = "kaz_Cyrl"
            if hasattr(tok, "lang_code_to_id"):
                forced_bos = tok.lang_code_to_id.get("eng_Latn")
            else:
                forced_bos = None

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        kwargs = {"max_new_tokens": 256, "num_beams": 4}
        if forced_bos is not None:
            kwargs["forced_bos_token_id"] = forced_bos

        with torch.no_grad():
            out_ids = mdl.generate(**inputs, **kwargs)
        translation = tok.decode(out_ids[0], skip_special_tokens=True).strip()
        if not translation:
            return text, False, None
        return translation, True, translation

    def _predict_transformer(self, text_en: str) -> tuple[float, float, int]:
        probs, token_len = self._predict_raw_probs(text_en)
        fake_prob = float(probs[self.fake_class_id])
        real_prob = float(probs[self.real_class_id])
        return fake_prob, real_prob, token_len

    def _predict_rf(self, text_original: str, language: str) -> Optional[float]:
        if self.rf_bundle is None:
            return None
        model = self.rf_bundle.get("model")
        feature_names = self.rf_bundle.get("feature_names")
        if model is None or not feature_names:
            return None

        feats = extract_meta_features(text_original, language)
        x = np.array([[feats.get(name, 0.0) for name in feature_names]], dtype=np.float32)
        proba = model.predict_proba(x)[0]
        # class 1 in the RF bundle is fake by training convention.
        return float(proba[1])

    @staticmethod
    def _rule_adjust_fake_probability(
        *,
        features: Dict[str, float],
        base_fake_prob: float,
        rf_prob: Optional[float],
    ) -> tuple[float, float]:
        # Lightweight safety layer: catches obvious fake-style patterns missed by the base model.
        up = 0.0
        down = 0.0

        clickbait_hits = features.get("clickbait_hits", 0.0)
        hedge_hits = features.get("hedge_hits", 0.0)
        source_hits = features.get("source_hits", 0.0)
        source_absence_hits = features.get("source_absence_hits", 0.0)
        uppercase_ratio = features.get("uppercase_ratio", 0.0)
        exclamation_density = features.get("exclamation_density", 0.0)
        all_caps_words = features.get("all_caps_words", 0.0)
        word_count = features.get("word_count", 0.0)

        if clickbait_hits >= 1:
            up += min(
                RULE_CFG["clickbait_cap"],
                RULE_CFG["clickbait_base"] + RULE_CFG["clickbait_step"] * float(clickbait_hits - 1),
            )
        if hedge_hits >= 1:
            up += min(
                RULE_CFG["hedge_cap"],
                RULE_CFG["hedge_base"] + RULE_CFG["hedge_step"] * float(hedge_hits - 1),
            )
        if source_absence_hits >= 1:
            up += min(
                RULE_CFG["absence_cap"],
                RULE_CFG["absence_base"] + RULE_CFG["absence_step"] * float(source_absence_hits - 1),
            )
        if source_hits == 0 and source_absence_hits == 0 and word_count >= 35:
            up += RULE_CFG["no_source_bonus"]
        if uppercase_ratio > 0.16 or all_caps_words >= 2:
            up += RULE_CFG["style_bonus"]
        if exclamation_density > 0.02:
            up += RULE_CFG["punc_bonus"]
        if source_absence_hits >= 2 and (clickbait_hits >= 1 or hedge_hits >= 1):
            up += RULE_CFG["strong_fake_combo_bonus"]

        if source_hits >= 2 and source_absence_hits == 0 and clickbait_hits == 0 and hedge_hits == 0:
            down += RULE_CFG["real_source_discount"]
        if rf_prob is not None and rf_prob < 0.2 and up < 0.2:
            down += RULE_CFG["rf_real_discount"]

        delta = float(np.clip(up - down, RULE_CFG["delta_min"], RULE_CFG["delta_max"]))
        adjusted = float(np.clip(base_fake_prob + delta, 0.0, 1.0))
        return adjusted, delta

    def analyze(self, *, text: str, language: str = "auto") -> AnalyzeResult:
        t0 = time.perf_counter()
        validation = ensure_valid_news_text(text)
        raw = self._normalize_text(validation.cleaned_text)
        detected_language = self.detect_language(raw, language)
        text_for_model, translated, translation_text = self._translate(raw, detected_language)
        fake_prob_trf, real_prob_trf, token_len = self._predict_transformer(text_for_model)
        rf_prob = self._predict_rf(raw, detected_language)

        if rf_prob is not None:
            # Ensemble smoothes edge cases and keeps RF baseline active.
            base_fake_prob = float(np.clip(0.8 * fake_prob_trf + 0.2 * rf_prob, 0.0, 1.0))
            model_trace = (
                f"transformer+rf|trf_fake={fake_prob_trf:.3f}|rf_fake={rf_prob:.3f}"
                f"|map=f{self.fake_class_id}"
            )
        else:
            base_fake_prob = fake_prob_trf
            model_trace = f"transformer|trf_fake={fake_prob_trf:.3f}|map=f{self.fake_class_id}"

        lang_for_rules = detected_language if detected_language in {"en", "ru", "kk"} else "en"
        rule_features = extract_meta_features(raw, lang_for_rules)
        fake_prob, rule_delta = self._rule_adjust_fake_probability(
            features=rule_features,
            base_fake_prob=base_fake_prob,
            rf_prob=rf_prob,
        )
        real_prob = 1.0 - fake_prob
        if abs(rule_delta) > 1e-9:
            model_trace += f"|rule_delta={rule_delta:+.3f}"

        label = "fake" if fake_prob >= real_prob else "real"

        report = build_xai_report(
            text_original=raw,
            detected_language=detected_language,
            fake_probability=fake_prob,
            rf_probability=rf_prob,
        )

        llm_explanation = generate_openai_explanation(
            original_text=raw,
            detected_language=detected_language,
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            xai_summary=report.detailed_summary,
            decision_path=report.decision_path,
            highlights=report.highlights,
            reasons=[
                {
                    "type": item.reason_type,
                    "text": item.text,
                    "weight": round(item.weight, 3),
                    "code": item.code,
                }
                for item in report.reasons
            ],
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        explanation = {
            "probability_fake_percent": round(fake_prob * 100.0, 2),
            "reasons": [
                {
                    "type": item.reason_type,
                    "text": item.text,
                    "weight": round(item.weight, 3),
                    "code": item.code,
                }
                for item in report.reasons
            ],
            "highlights": report.highlights,
            "compact_codes": report.compact_codes,
            "agent_hint_min_tokens": short_signal_prompt_lang(detected_language),
            "detailed_summary": report.detailed_summary,
            "llm_detailed_summary": llm_explanation.text,
            "llm_model": llm_explanation.model,
            "llm_latency_ms": llm_explanation.latency_ms,
            "llm_error": llm_explanation.error,
            "decision_path": report.decision_path,
            "feature_snapshot": report.feature_snapshot,
        }
        return AnalyzeResult(
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            detected_language=detected_language,
            translated_to_english=translated,
            translation_text=translation_text,
            latency_ms=latency_ms,
            explanation=explanation,
            model_trace=model_trace,
            token_length=token_len,
        )
