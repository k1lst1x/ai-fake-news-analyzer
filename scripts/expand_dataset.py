from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd
from datasets import load_dataset


RANDOM_SEED = 42
MIN_TEXT_LEN = 80
MAX_TEXT_LEN = 7000

CLICKBAIT_PREFIX = [
    "SHOCKING UPDATE",
    "URGENT EXCLUSIVE",
    "SENSATIONAL REVEAL",
    "YOU WON'T BELIEVE THIS",
    "BREAKING TRUTH",
]

FAKE_STYLE_WORDS = [
    "shocking",
    "sensational",
    "exclusive",
    "breaking",
    "secret",
    "you won't believe",
    "urgent",
    "fake",
    "satire",
    "hoax",
]


def clean_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def valid_text(text: str) -> bool:
    return MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN


def size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024**3)


def init_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "lang", "source", "synthetic"])
        writer.writeheader()


def append_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "lang", "source", "synthetic"])
        writer.writerows(rows)


def load_local_isot(data_dir: Path) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")

    fake_rows = []
    for txt in fake_df["text"].astype(str):
        t = clean_text(txt)
        if valid_text(t):
            fake_rows.append({"text": t, "label": "1", "lang": "en", "source": "isot_fake", "synthetic": "0"})

    real_rows = []
    for txt in true_df["text"].astype(str):
        t = clean_text(txt)
        if valid_text(t):
            real_rows.append({"text": t, "label": "0", "lang": "en", "source": "isot_true", "synthetic": "0"})
    return fake_rows, real_rows


def _fake_style_score(text: str) -> float:
    low = clean_text(text).lower()
    if not low:
        return 0.0
    kw_hits = sum(1 for w in FAKE_STYLE_WORDS if w in low)
    punct = low.count("!") * 0.15 + low.count("?") * 0.08
    caps = sum(1 for t in low.split() if t.isupper() and len(t) >= 4) * 0.2
    return kw_hits + punct + caps


def load_hf_labeled_dataset(
    dataset_name: str,
    *,
    text_fields: List[str],
    label_field: str,
    max_rows: int,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ds_dict = load_dataset(dataset_name)

    sample_scores: Dict[str, List[float]] = {}
    scanned = 0
    for split_name, split in ds_dict.items():
        for row in split:
            text = ""
            for field in text_fields:
                val = row.get(field)
                if val is not None and str(val).strip():
                    text = str(val)
                    break
            text = clean_text(text)
            if not valid_text(text):
                continue
            label_val = str(row.get(label_field))
            sample_scores.setdefault(label_val, []).append(_fake_style_score(text))
            scanned += 1
            if scanned >= 4000:
                break
        if scanned >= 4000:
            break

    if len(sample_scores) < 2:
        raise RuntimeError(f"{dataset_name}: not enough label diversity in {label_field}")

    avg_by_label = {
        label: (sum(vals) / max(len(vals), 1))
        for label, vals in sample_scores.items()
    }
    fake_label_val = max(avg_by_label.items(), key=lambda x: x[1])[0]

    fake_rows: List[Dict[str, str]] = []
    real_rows: List[Dict[str, str]] = []
    for split_name, split in ds_dict.items():
        for row in split:
            text = ""
            for field in text_fields:
                val = row.get(field)
                if val is not None and str(val).strip():
                    text = str(val)
                    break
            text = clean_text(text)
            if not valid_text(text):
                continue
            label_val = str(row.get(label_field))
            mapped_label = "1" if label_val == fake_label_val else "0"
            payload = {
                "text": text,
                "label": mapped_label,
                "lang": "en",
                "source": dataset_name.replace("/", "_"),
                "synthetic": "0",
            }
            if mapped_label == "1":
                fake_rows.append(payload)
            else:
                real_rows.append(payload)
            if len(fake_rows) + len(real_rows) >= max_rows:
                break
        if len(fake_rows) + len(real_rows) >= max_rows:
            break
    return fake_rows, real_rows


def iter_hf_cc_news(max_rows: int) -> Iterator[Dict[str, str]]:
    # Large real-news source.
    dataset = load_dataset("cc_news", split="train", streaming=True)
    count = 0
    for row in dataset:
        text = clean_text(row.get("text", ""))
        if not valid_text(text):
            continue
        yield {"text": text, "label": "0", "lang": "en", "source": "cc_news", "synthetic": "0"}
        count += 1
        if count >= max_rows:
            break


def iter_hf_ag_news(max_rows: int) -> Iterator[Dict[str, str]]:
    dataset = load_dataset("ag_news", split="train")
    count = 0
    for row in dataset:
        text = clean_text(row.get("text", ""))
        if not valid_text(text):
            continue
        yield {"text": text, "label": "0", "lang": "en", "source": "ag_news", "synthetic": "0"}
        count += 1
        if count >= max_rows:
            break


def synthesize_fake_from_real(text: str) -> str:
    prefix = random.choice(CLICKBAIT_PREFIX)
    tail = random.choice(
        [
            "Experts hide this from you.",
            "The real story is being censored.",
            "Share before it gets deleted.",
            "Official media won't show this.",
        ]
    )
    text = clean_text(text)
    base = text[:1500]
    return f"{prefix}! {base} {tail}"


def build_dataset(
    *,
    data_dir: Path,
    out_csv: Path,
    target_gb: float,
    max_cc_rows: int,
    max_ag_rows: int,
    max_extra_fake_rows: int,
    synth_ratio: float,
) -> None:
    random.seed(RANDOM_SEED)
    init_output(out_csv)

    seen = set()
    fake_count = 0
    real_count = 0

    def write(rows: Iterable[Dict[str, str]]) -> None:
        nonlocal fake_count, real_count
        buffer: List[Dict[str, str]] = []
        for row in rows:
            key = row["text"][:256]
            if key in seen:
                continue
            seen.add(key)
            buffer.append(row)
            if row["label"] == "1":
                fake_count += 1
            else:
                real_count += 1
            if len(buffer) >= 5000:
                append_rows(out_csv, buffer)
                buffer = []
        append_rows(out_csv, buffer)

    print("[1/5] Loading local ISOT...")
    fake_rows, real_rows = load_local_isot(data_dir)
    write(fake_rows)
    write(real_rows)
    print(f"  rows: fake={fake_count}, real={real_count}, size={size_gb(out_csv):.2f} GB")

    print("[2/5] Loading HF fake-news datasets...")
    hf_sources = [
        ("GonzaloA/fake_news", ["text", "title"], "label"),
        ("mrm8488/fake-news", ["text"], "label"),
        ("ikekobby/40-percent-cleaned-preprocessed-fake-real-news", ["clean_article", "article"], "label"),
    ]
    for ds_name, text_fields, label_field in hf_sources:
        try:
            fake_rows_hf, real_rows_hf = load_hf_labeled_dataset(
                ds_name,
                text_fields=text_fields,
                label_field=label_field,
                max_rows=max_extra_fake_rows,
            )
            write(fake_rows_hf)
            write(real_rows_hf)
            print(
                f"  loaded {ds_name}: +fake={len(fake_rows_hf)} +real={len(real_rows_hf)}"
            )
        except Exception as exc:
            print(f"  skipped {ds_name}: {exc}")
    print(f"  rows: fake={fake_count}, real={real_count}, size={size_gb(out_csv):.2f} GB")

    print("[3/5] Loading HF AG News (real)...")
    try:
        write(iter_hf_ag_news(max_rows=max_ag_rows))
    except Exception as exc:
        print(f"  skipped AG News: {exc}")
    print(f"  rows: fake={fake_count}, real={real_count}, size={size_gb(out_csv):.2f} GB")

    print("[4/5] Streaming HF CC News (real + synthetic fake)...")
    try:
        chunk: List[Dict[str, str]] = []
        for row in iter_hf_cc_news(max_rows=max_cc_rows):
            chunk.append(row)
            if random.random() < synth_ratio:
                synth_text = synthesize_fake_from_real(row["text"])
                if valid_text(synth_text):
                    chunk.append(
                        {
                            "text": synth_text,
                            "label": "1",
                            "lang": row["lang"],
                            "source": "synthetic_clickbait_from_cc_news",
                            "synthetic": "1",
                        }
                    )
            if len(chunk) >= 4000:
                write(chunk)
                chunk = []
                if size_gb(out_csv) >= target_gb:
                    break
        if chunk:
            write(chunk)
    except Exception as exc:
        print(f"  skipped CC News: {exc}")

    print(f"  rows: fake={fake_count}, real={real_count}, size={size_gb(out_csv):.2f} GB")

    print("[5/5] Balancing classes with synthetic fake if needed...")
    if fake_count < real_count * 0.75:
        deficit = int(real_count * 0.75 - fake_count)
        print(f"  need ~{deficit} synthetic fake rows")
        reader = pd.read_csv(out_csv, chunksize=25000)
        for chunk_df in reader:
            labels = chunk_df["label"].astype(str)
            real_texts = chunk_df[labels == "0"]["text"].astype(str).tolist()
            synth_rows = []
            for t in real_texts:
                if deficit <= 0:
                    break
                synth = synthesize_fake_from_real(t)
                if not valid_text(synth):
                    continue
                key = synth[:256]
                if key in seen:
                    continue
                seen.add(key)
                synth_rows.append(
                    {
                        "text": synth,
                        "label": "1",
                        "lang": "en",
                        "source": "synthetic_balance",
                        "synthetic": "1",
                    }
                )
                deficit -= 1
                fake_count += 1
            append_rows(out_csv, synth_rows)
            if deficit <= 0 or size_gb(out_csv) >= target_gb:
                break

    final_size = size_gb(out_csv)
    print("\n=== DONE ===")
    print(f"Output file: {out_csv}")
    print(f"Final size: {final_size:.2f} GB")
    print(f"Rows: fake={fake_count}, real={real_count}, total={fake_count + real_count}")
    print(
        "Note: for strict 15-20GB target, increase --max-cc-rows and/or --target-gb "
        "and rerun overnight."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand fake-news dataset to large-scale CSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/expanded/expanded_news_dataset.csv"))
    parser.add_argument("--target-gb", type=float, default=15.0)
    parser.add_argument("--max-cc-rows", type=int, default=5_000_000)
    parser.add_argument("--max-ag-rows", type=int, default=200_000)
    parser.add_argument("--max-extra-fake-rows", type=int, default=180_000)
    parser.add_argument(
        "--synth-ratio",
        type=float,
        default=0.35,
        help="Probability of generating synthetic fake sample from each real CC News text.",
    )
    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        out_csv=args.out_csv,
        target_gb=args.target_gb,
        max_cc_rows=args.max_cc_rows,
        max_ag_rows=args.max_ag_rows,
        max_extra_fake_rows=args.max_extra_fake_rows,
        synth_ratio=args.synth_ratio,
    )


if __name__ == "__main__":
    main()
