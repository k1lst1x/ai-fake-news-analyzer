import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


# ---------- LOAD ENV ----------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

APP_TITLE = os.getenv("APP_TITLE", "AI Fake News Analyzer")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models/fake_news_model"))
RU_EN_DIR = Path(os.getenv("RU_EN_DIR", "./models/ru_en"))
KK_EN_DIR = Path(os.getenv("KK_EN_DIR", "./models/nllb"))

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
FAKE_CLASS_ID = int(os.getenv("FAKE_CLASS_ID", "0"))
REAL_CLASS_ID = int(os.getenv("REAL_CLASS_ID", "1"))

MODEL_DIR = (BASE_DIR / MODEL_DIR).resolve()
RU_EN_DIR = (BASE_DIR / RU_EN_DIR).resolve()
KK_EN_DIR = (BASE_DIR / KK_EN_DIR).resolve()


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------- STYLES ----------
st.markdown("""
<style>
body {
    background-color: #1E1F23;
}
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 3rem;
}
h1, h2, h3, h4 {
    color: #E6E6E6;
    text-align: center;
}
p, span, label {
    color: #A0A0A0;
    text-align: center;
}
textarea {
    background-color: #2B2C31 !important;
    color: #E6E6E6 !important;
    border-radius: 12px !important;
}

.card {
    background-color: #2B2C31;
    padding: 24px;
    border-radius: 16px;
    margin-top: 24px;
    text-align: left;
}
.fake {
    border-left: 6px solid #E57373;
}
.real {
    border-left: 6px solid #81C784;
}
.stButton {
    display: flex;
    justify-content: center;
    margin-top: 16px;
}

.stButton > button {
    background-color: #5F6F52;
    color: #F1F1F1;
    border-radius: 10px;
    padding: 0.7rem 1.6rem;
    border: 1px solid #6E7F5F;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #556348;
}
</style>
""", unsafe_allow_html=True)


# ---------- HERO ----------
st.markdown("<h1>Fake News Detection with Explainable AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>An AI-powered system that analyzes news articles and estimates the probability of misinformation.</p>",
    unsafe_allow_html=True
)

news_text = st.text_area(
    label="News article text",
    height=220,
    placeholder="Paste a news article here for analysis...",
    label_visibility="collapsed"
)


# ---------- UTILS ----------
def has_cyrillic(s: str) -> bool:
    s = s or ""
    for ch in s:
        c = ord(ch)
        if 0x0400 <= c <= 0x04FF:
            return True
    return False


def folder_has_config(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_classifier():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.to(device).eval()
    return tokenizer, model, device


@st.cache_resource
def load_translator_ru_en():
    device = get_device()
    if not folder_has_config(RU_EN_DIR):
        return None, None, device
    tok = AutoTokenizer.from_pretrained(
        str(RU_EN_DIR),
        local_files_only=True,
        use_fast=False
    )
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(RU_EN_DIR), local_files_only=True)
    mdl.to(device).eval()
    return tok, mdl, device


@st.cache_resource
def load_translator_kk_en():
    device = get_device()
    if not folder_has_config(KK_EN_DIR):
        return None, None, device
    tok = AutoTokenizer.from_pretrained(
        str(KK_EN_DIR),
        local_files_only=True,
        use_fast=False
    )
    mdl = AutoModelForSeq2SeqLM.from_pretrained(str(KK_EN_DIR), local_files_only=True)
    mdl.to(device).eval()
    return tok, mdl, device


clf_tok, clf_model, device = load_classifier()
ru_en_tok, ru_en_model, _ = load_translator_ru_en()
kk_en_tok, kk_en_model, _ = load_translator_kk_en()


# ---------- TRANSLATION ----------
def translate_ru_to_en(text: str) -> str:
    if ru_en_tok is None or ru_en_model is None:
        return ""

    forced_bos_token_id = None
    if hasattr(ru_en_tok, "lang_code_to_id"):
        if hasattr(ru_en_tok, "src_lang"):
            ru_en_tok.src_lang = "rus_Cyrl"
        forced_bos_token_id = ru_en_tok.lang_code_to_id.get("eng_Latn")

    inputs = ru_en_tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": 256, "num_beams": 4}
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

    with torch.no_grad():
        out_ids = ru_en_model.generate(**inputs, **gen_kwargs)

    return ru_en_tok.decode(out_ids[0], skip_special_tokens=True)


def translate_kk_to_en(text: str) -> str:
    if kk_en_tok is None or kk_en_model is None:
        return ""

    if hasattr(kk_en_tok, "src_lang"):
        kk_en_tok.src_lang = "kaz_Cyrl"

    forced_bos_token_id = None
    if hasattr(kk_en_tok, "lang_code_to_id"):
        forced_bos_token_id = kk_en_tok.lang_code_to_id.get("eng_Latn")

    inputs = kk_en_tok(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": 256, "num_beams": 4}
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

    with torch.no_grad():
        out_ids = kk_en_model.generate(**inputs, **gen_kwargs)

    return kk_en_tok.decode(out_ids[0], skip_special_tokens=True)


# ---------- PREDICT ----------
def predict_fake_real(text: str):
    inputs = clf_tok(text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = clf_model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

    fake_prob = float(probs[FAKE_CLASS_ID].item())
    real_prob = float(probs[REAL_CLASS_ID].item())

    label = "fake" if fake_prob >= real_prob else "real"
    return label, fake_prob, real_prob


# ---------- LANGUAGE ----------
lang = st.selectbox("Input language", ["English", "Russian", "Kazakh"], index=0)


clicked = st.button("Analyze News")


# ---------- RESULT ----------
if clicked and news_text.strip():
    with st.spinner("Analyzing with ML model..."):
        used_translation = False
        translated_text = ""
        text_for_model = news_text.strip()

        if lang == "Russian":
            used_translation = True
            translated_text = translate_ru_to_en(text_for_model)
            if translated_text.strip():
                text_for_model = translated_text

        elif lang == "Kazakh":
            used_translation = True
            translated_text = translate_kk_to_en(text_for_model)
            if translated_text.strip():
                text_for_model = translated_text

        else:
            if has_cyrillic(text_for_model):
                used_translation = True
                translated_text = translate_ru_to_en(text_for_model)
                if translated_text.strip():
                    text_for_model = translated_text

        label, fake_p, real_p = predict_fake_real(text_for_model)

    css_class = "fake" if label == "fake" else "real"
    title = "LIKELY FAKE NEWS" if label == "fake" else "LIKELY REAL NEWS"
    color = "#E57373" if label == "fake" else "#81C784"

    st.markdown(f"""
    <div class="card {css_class}">
        <h3 style="color:{color};">{title}</h3>
        <p><strong>Fake probability:</strong> {fake_p * 100:.1f}%</p>
        <p><strong>Real probability:</strong> {real_p * 100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    if used_translation and translated_text.strip():
        label_text = "RU → EN" if lang == "Russian" else ("KK → EN" if lang == "Kazakh" else "Cyrillic → EN")
        st.markdown(f"""
        <div class="card">
            <h4>Translation ({label_text}) used for the model</h4>
            <p style="text-align:left; color:#A0A0A0; white-space:pre-wrap;">{translated_text}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h4>Model Explanation</h4>
        <p>
        Transformer-based model predicts misinformation probability.
        <br/>
        <span style="opacity:0.85;">Mapping: class {FAKE_CLASS_ID} = fake, class {REAL_CLASS_ID} = real</span>
        <br/>
        <span style="opacity:0.85;">Device: {device}</span>
        <br/>
        <span style="opacity:0.85;">RU/KK support: via local translation to English</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------- FOOTER ----------
st.markdown(
    "<p style='margin-top:40px; font-size:12px; text-align:center;'>AI Fake News Analyzer • MVP Demo</p>",
    unsafe_allow_html=True
)
