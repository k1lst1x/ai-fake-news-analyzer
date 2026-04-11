# train_eval.py
from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import torch
from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAKE_FILENAME = "Fake.csv"
TRUE_FILENAME = "True.csv"

MODEL_DIR = BASE_DIR / "models" / "fake_news_model"

MAX_TOKENS = 512

# Р§С‚РѕР±С‹ РЅРµ Р¶РґР°С‚СЊ РІРµС‡РЅРѕСЃС‚СЊ: РїРѕ СЃРєРѕР»СЊРєРѕ СЃС‚СЂРѕРє Р±СЂР°С‚СЊ РёР· РєР°Р¶РґРѕРіРѕ РєР»Р°СЃСЃР° (None = РІСЃРµ)
LIMIT_PER_CLASS: Optional[int] = 5000

TEST_SIZE = 0.2
RANDOM_STATE = 42

PIPELINE_BATCH_SIZE = 16
PROGRESS_CHUNK_STEP = 128

# Р”Р»СЏ "РїСЂРѕРІРµСЂРєРё РЅР° РёСЃС‚РѕС‡РЅРёРєРё" (Reuters-РґРµС‚РµРєС‚РѕСЂ): СЃСЂРµР·Р°РµРј РїРµСЂРІС‹Рµ N СЃРёРјРІРѕР»РѕРІ
HEADER_CUT_CHARS = 200


# =========================
# UTILS
# =========================

def die(msg: str) -> None:
    print(f"\n[ERROR] {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def pick_text_column(df: pd.DataFrame) -> str:
    candidates = ["text", "content", "article", "body", "news"]
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    die(f"РќРµ РЅР°Р№РґРµРЅ СЃС‚РѕР»Р±РµС† СЃ С‚РµРєСЃС‚РѕРј. РљРѕР»РѕРЅРєРё: {list(df.columns)}")


def load_csv(path: Path, y_value: int) -> pd.DataFrame:
    if not path.exists():
        die(f"Р¤Р°Р№Р» РЅРµ РЅР°Р№РґРµРЅ: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        die(f"РќРµ РјРѕРіСѓ РїСЂРѕС‡РёС‚Р°С‚СЊ CSV {path}: {e}")

    if df.empty:
        die(f"РџСѓСЃС‚РѕР№ CSV: {path}")

    text_col = pick_text_column(df)

    out = df[[text_col]].copy()
    out.rename(columns={text_col: "text"}, inplace=True)

    out["text"] = out["text"].astype(str).fillna("")
    out["text"] = out["text"].str.replace("\r", " ").str.replace("\n", " ").str.strip()
    out = out[out["text"].str.len() > 0].copy()

    out["y"] = y_value
    return out


def get_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def flatten_outputs(outputs: Any) -> List[Dict[str, Any]]:
    if not outputs:
        return []
    if isinstance(outputs[0], list):
        return [x[0] for x in outputs]
    return outputs


def infer_mapping(sample_outputs: List[Dict[str, Any]], sample_y: List[int]) -> str:
    labels = [o["label"] for o in sample_outputs]

    # A: LABEL_1 = fake
    pred_a = [1 if lbl == "LABEL_1" else 0 for lbl in labels]
    acc_a = accuracy_score(sample_y, pred_a)

    # B: LABEL_0 = fake
    pred_b = [1 if lbl == "LABEL_0" else 0 for lbl in labels]
    acc_b = accuracy_score(sample_y, pred_b)

    return "LABEL_1_IS_FAKE" if acc_a >= acc_b else "LABEL_0_IS_FAKE"


def apply_mapping(outputs: List[Dict[str, Any]], mapping: str) -> List[int]:
    if mapping == "LABEL_1_IS_FAKE":
        return [1 if o["label"] == "LABEL_1" else 0 for o in outputs]
    return [1 if o["label"] == "LABEL_0" else 0 for o in outputs]


def run_inference(
    clf,
    texts: List[str],
    *,
    max_tokens: int,
    batch_size: int,
    chunk_step: int
) -> List[Dict[str, Any]]:
    outputs_all: List[Dict[str, Any]] = []
    for i in tqdm(range(0, len(texts), chunk_step), desc="Inference", unit="chunk"):
        batch_texts = texts[i:i + chunk_step]
        out = clf(
            batch_texts,
            truncation=True,
            max_length=max_tokens,
            batch_size=batch_size
        )
        out = flatten_outputs(out)
        outputs_all.extend(out)
    return outputs_all


def print_metrics(y_true: List[int], y_pred: List[int], title: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["real", "fake"])

    print(f"\n================ {title} ================")
    print(f"Accuracy: {acc:.6f}")
    print("Confusion matrix (rows=true, cols=pred) [real,fake]:")
    print(cm)
    print("\nClassification report:")
    print(report)


# =========================
# MAIN
# =========================

def main() -> None:
    fake_path = DATA_DIR / FAKE_FILENAME
    true_path = DATA_DIR / TRUE_FILENAME

    if not MODEL_DIR.exists():
        die(f"РњРѕРґРµР»СЊ РЅРµ РЅР°Р№РґРµРЅР°: {MODEL_DIR.resolve()}\nРЎРЅР°С‡Р°Р»Р° Р·Р°РїСѓСЃС‚Рё: python download_model.py")

    if not fake_path.exists():
        die(f"РќРµ РЅР°Р№РґРµРЅ Fake.csv: {fake_path}")
    if not true_path.exists():
        die(f"РќРµ РЅР°Р№РґРµРЅ True.csv: {true_path}")

    print("[INFO] Loading datasets...")
    fake_df = load_csv(fake_path, y_value=1)
    true_df = load_csv(true_path, y_value=0)

    if LIMIT_PER_CLASS is not None:
        fake_df = fake_df.sample(n=min(LIMIT_PER_CLASS, len(fake_df)), random_state=RANDOM_STATE)
        true_df = true_df.sample(n=min(LIMIT_PER_CLASS, len(true_df)), random_state=RANDOM_STATE)

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if df.empty:
        die("РџРѕСЃР»Рµ РѕС‡РёСЃС‚РєРё РЅРµ РѕСЃС‚Р°Р»РѕСЃСЊ РґР°РЅРЅС‹С….")

    print(f"[INFO] Total: {len(df)} | fake={int(df['y'].sum())} | real={int((df['y']==0).sum())}")

    # split
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["y"]
    )
    print(f"[INFO] Test size: {len(test)}")

    # LEAK CHECK: exact duplicates train/test
    train_set = set(train["text"].tolist())
    test_set = set(test["text"].tolist())
    overlap = len(train_set & test_set)
    print(f"[LEAK] Train/Test exact text overlap: {overlap}")

    texts = test["text"].tolist()
    y_true = test["y"].tolist()

    # GPU
    device = get_device()
    if device == 0:
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU (CUDA not available)")

    clf = pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=device
    )

    print("[INFO] Running inference...")
    outputs_all = run_inference(
        clf,
        texts,
        max_tokens=MAX_TOKENS,
        batch_size=PIPELINE_BATCH_SIZE,
        chunk_step=PROGRESS_CHUNK_STEP
    )

    if len(outputs_all) != len(texts):
        die(f"outputs != texts: {len(outputs_all)} vs {len(texts)}")

    # mapping
    sample_n = min(500, len(outputs_all))
    mapping = infer_mapping(outputs_all[:sample_n], y_true[:sample_n])
    print(f"[INFO] Label mapping: {mapping}")

    y_pred = apply_mapping(outputs_all, mapping)

    # metrics (main)
    print_metrics(y_true, y_pred, "RESULTS")

    # SANITY CHECK: shuffled labels should drop to ~0.5
    y_shuffled = y_true[:]
    random.Random(42).shuffle(y_shuffled)
    sanity_acc = accuracy_score(y_shuffled, y_pred)
    print(f"[SANITY] Accuracy with shuffled labels: {sanity_acc:.6f}")

    # HEADER CHECK: cut first N chars and evaluate again (helps detect "Reuters header" leakage)
    print(f"\n[CHECK] Re-evaluating after cutting first {HEADER_CUT_CHARS} chars from each text...")
    cut_texts = [t[HEADER_CUT_CHARS:] if len(t) > HEADER_CUT_CHARS else t for t in texts]

    outputs_cut = run_inference(
        clf,
        cut_texts,
        max_tokens=MAX_TOKENS,
        batch_size=PIPELINE_BATCH_SIZE,
        chunk_step=PROGRESS_CHUNK_STEP
    )
    outputs_cut = outputs_cut[:len(cut_texts)]
    y_pred_cut = apply_mapping(outputs_cut, mapping)
    print_metrics(y_true, y_pred_cut, "RESULTS_AFTER_HEADER_CUT")

    # Mistake examples (real ones)
    print("\n[INFO] Mistake examples (up to 5):")
    wrong = 0
    for idx, (t, yt, yp) in enumerate(zip(texts, y_true, y_pred)):
        if yt != yp:
            print(f"- idx={idx} true={'fake' if yt==1 else 'real'} pred={'fake' if yp==1 else 'real'} | {t[:180]!r}")
            wrong += 1
            if wrong >= 5:
                break


if __name__ == "__main__":
    main()

