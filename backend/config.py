from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def _load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _getenv(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


_load_local_env()

DEFAULT_MODEL_DIR = MODELS_DIR / "fake_news_model_ft"
if not DEFAULT_MODEL_DIR.exists():
    DEFAULT_MODEL_DIR = MODELS_DIR / "fake_news_model"
MODEL_DIR = Path(_getenv("FAKE_NEWS_MODEL_DIR", "MODEL_DIR", default=str(DEFAULT_MODEL_DIR)))
RU_EN_DIR = Path(_getenv("RU_EN_DIR", default=str(MODELS_DIR / "ru_en")))
KK_EN_DIR = Path(_getenv("KK_EN_DIR", default=str(MODELS_DIR / "nllb")))
RF_MODEL_PATH = Path(_getenv("RF_MODEL_PATH", default=str(MODELS_DIR / "rf_meta_model.joblib")))

ANALYTICS_DB_PATH = Path(_getenv("ANALYTICS_DB_PATH", default=str(DATA_DIR / "analytics.db")))

MAX_MODEL_TOKENS = int(_getenv("MAX_MODEL_TOKENS", "MAX_TOKENS", default="512"))
FAKE_CLASS_ID = int(_getenv("FAKE_CLASS_ID", default="0"))
REAL_CLASS_ID = int(_getenv("REAL_CLASS_ID", default="1"))
OPENAI_API_KEY = _getenv("OPENAI_API_KEY", default="")
OPENAI_EXPLANATION_MODEL = _getenv("OPENAI_EXPLANATION_MODEL", default="gpt-5.4-mini")
OPENAI_EXPLANATION_TIMEOUT = float(_getenv("OPENAI_EXPLANATION_TIMEOUT", default="25"))
OPENAI_EXPLANATION_REASONING = _getenv("OPENAI_EXPLANATION_REASONING", default="low")
OPENAI_EXPLANATION_ENABLED = _getenv("OPENAI_EXPLANATION_ENABLED", default="1").lower() not in {
    "0",
    "false",
    "off",
    "no",
}

# Hard cap protects latency and lowers token usage in inference/translation.
MAX_INPUT_CHARS = int(_getenv("MAX_INPUT_CHARS", default="4500"))
