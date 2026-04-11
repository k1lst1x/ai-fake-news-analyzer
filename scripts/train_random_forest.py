from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.xai import extract_meta_features


def clean_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path, sample_size: int | None = None, chunksize: int = 120_000) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    required = {"text", "label"}
    sample_size = int(sample_size or 350_000)

    target_fake = sample_size // 2
    target_real = sample_size - target_fake
    picked_fake = 0
    picked_real = 0
    selected_parts: List[pd.DataFrame] = []

    for idx, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        if not required.issubset(chunk.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")

        chunk = chunk[["text", "label"]].copy()
        chunk["text"] = chunk["text"].astype(str).map(clean_text)
        chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(int)
        chunk = chunk[chunk["text"].str.len() >= 30]
        if chunk.empty:
            continue

        need_fake = max(0, target_fake - picked_fake)
        if need_fake > 0:
            fake_part = chunk[chunk["label"] == 1]
            if not fake_part.empty:
                take_n = min(need_fake, len(fake_part))
                selected = fake_part.sample(n=take_n, random_state=42 + idx)
                selected_parts.append(selected)
                picked_fake += take_n

        need_real = max(0, target_real - picked_real)
        if need_real > 0:
            real_part = chunk[chunk["label"] == 0]
            if not real_part.empty:
                take_n = min(need_real, len(real_part))
                selected = real_part.sample(n=take_n, random_state=42000 + idx)
                selected_parts.append(selected)
                picked_real += take_n

        if idx % 10 == 0:
            print(
                f"[LOAD] chunk={idx} picked_fake={picked_fake}/{target_fake} "
                f"picked_real={picked_real}/{target_real}"
            )
        if picked_fake >= target_fake and picked_real >= target_real:
            break

    if not selected_parts:
        raise RuntimeError("No rows selected from dataset.")

    df = pd.concat(selected_parts, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"[LOAD] selected rows after dedup: {len(df)}")
    return df


def build_feature_table(df: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    rows: List[Dict[str, float]] = []
    for text in df["text"].tolist():
        rows.append(extract_meta_features(text, "en"))
    feat_df = pd.DataFrame(rows).fillna(0.0)
    return feat_df.to_numpy(dtype=np.float32), feat_df.columns.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF meta-model over linguistic features.")
    parser.add_argument("--data-csv", type=Path, default=Path("data/expanded/expanded_news_dataset.csv"))
    parser.add_argument("--sample-size", type=int, default=350_000)
    parser.add_argument("--chunksize", type=int, default=120_000)
    parser.add_argument("--model-out", type=Path, default=Path("models/rf_meta_model.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("models/rf_meta_metrics.json"))
    args = parser.parse_args()

    print(f"[INFO] Loading data: {args.data_csv}")
    df = load_dataset(args.data_csv, sample_size=args.sample_size, chunksize=args.chunksize)
    print(f"[INFO] Rows after cleaning: {len(df)}")

    X, feature_names = build_feature_table(df)
    y = df["label"].to_numpy(dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=420,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    print("[INFO] Training Random Forest...")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred))
    report = classification_report(y_test, pred, digits=4)
    print(f"[RESULT] accuracy={acc:.4f} f1={f1:.4f}")
    print(report)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "feature_names": feature_names,
        "meta": {"class_1": "fake", "class_0": "real"},
    }
    joblib.dump(bundle, args.model_out)
    print(f"[INFO] Saved model: {args.model_out}")

    importances = dict(sorted(zip(feature_names, clf.feature_importances_), key=lambda x: x[1], reverse=True))
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "n_rows": int(len(df)),
        "feature_importance_top10": {k: float(v) for k, v in list(importances.items())[:10]},
    }
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
