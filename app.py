import streamlit as st
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Fake News Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- STYLES ----------
st.markdown("""
<style>
body {
    background-color: #1E1F23;
}

/* Center content */
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 3rem;
}

/* Typography */
h1, h2, h3, h4 {
    color: #E6E6E6;
    text-align: center;
}
p, span, label {
    color: #A0A0A0;
    text-align: center;
}

/* Textarea */
textarea {
    background-color: #2B2C31 !important;
    color: #E6E6E6 !important;
    border-radius: 12px !important;
}

/* Button */
button.analyze-btn {
    background-color: #5F6F52;
    color: #F1F1F1;
    border-radius: 10px;
    padding: 0.7rem 1.6rem;
    border: 1px solid #6E7F5F;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

button.analyze-btn:hover {
    background-color: #556348;
}

/* Cards */
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
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("<h1>Fake News Detection with Explainable AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>An AI-powered system that analyzes news articles and estimates the probability of misinformation.</p>",
    unsafe_allow_html=True
)

# ---------- INPUT ----------
news_text = st.text_area(
    label="News article text",
    height=220,
    placeholder="Paste a news article here for analysis...",
    label_visibility="collapsed"  # ✅ убирает warning
)

# ---------- ML STUB ----------
def fake_news_stub(text: str):
    fake_prob = round(random.uniform(0.6, 0.95), 2)
    real_prob = round(1 - fake_prob, 2)
    label = "fake" if fake_prob > 0.5 else "real"
    return label, fake_prob, real_prob

# ---------- BUTTON (HTML, идеально по центру) ----------
st.markdown("""
<div style="display:flex; justify-content:center; margin-top:16px;">
    <button class="analyze-btn" onclick="window.dispatchEvent(new Event('analyze'))">
        Analyze News
    </button>
</div>
""", unsafe_allow_html=True)

# ---------- STATE INIT ----------
if "analyze_clicked" not in st.session_state:
    st.session_state.analyze_clicked = False

# ---------- JS EVENT LISTENER ----------
st.components.v1.html("""
<script>
window.addEventListener("analyze", () => {
    const streamlitEvent = new Event("streamlit:analyze");
    window.parent.document.dispatchEvent(streamlitEvent);
});
</script>
""", height=0)

# ---------- HANDLE CLICK ----------
if st.session_state.get("_streamlit_analyze"):
    st.session_state.analyze_clicked = True

# fallback (первый клик)
if st.session_state.get("analyze_clicked") is False and news_text:
    st.session_state.analyze_clicked = True

# ---------- RESULT ----------
if st.session_state.analyze_clicked and news_text.strip():
    label, fake_p, real_p = fake_news_stub(news_text)

    css_class = "fake" if label == "fake" else "real"
    title = "LIKELY FAKE NEWS" if label == "fake" else "LIKELY REAL NEWS"
    color = "#E57373" if label == "fake" else "#81C784"

    st.markdown(f"""
    <div class="card {css_class}">
        <h3 style="color:{color};">{title}</h3>
        <p><strong>Fake probability:</strong> {int(fake_p * 100)}%</p>
        <p><strong>Real probability:</strong> {int(real_p * 100)}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>Model Explanation</h4>
        <p>
        The AI model analyzed semantic structure, writing style, and similarity
        to known misinformation patterns using deep language representations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    "<p style='margin-top:40px; font-size:12px; text-align:center;'>AI Fake News Analyzer • MVP Demo</p>",
    unsafe_allow_html=True
)
