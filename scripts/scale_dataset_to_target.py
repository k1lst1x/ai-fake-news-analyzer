from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


FAKE_PREFIXES = [
    "Shocking update:",
    "Urgent exclusive:",
    "Breaking claim:",
    "Unconfirmed report:",
    "Leaked information:",
]

FAKE_SUFFIXES = [
    "Share quickly before deletion.",
    "Official media allegedly ignores this.",
    "Readers report unusual inconsistencies.",
    "Fact-check pending due contradictory sources.",
]

REAL_PREFIXES = [
    "News digest:",
    "Agency report:",
    "Public statement:",
    "Verified update:",
]

REAL_SUFFIXES = [
    "Source references available in public records.",
    "Official commentary attached.",
    "Reported with standard editorial checks.",
]


def clean_text(text: str) -> str:
    txt = str(text or "")
    txt = txt.replace("\r", " ").replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024**3)


def write_header(path: Path) -> None:
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


def augment_text(text: str, label: int, cycle: int, row_idx: int) -> str:
    text = clean_text(text)
    if label == 1:
        pref = random.choice(FAKE_PREFIXES)
        suff = random.choice(FAKE_SUFFIXES)
    else:
        pref = random.choice(REAL_PREFIXES)
        suff = random.choice(REAL_SUFFIXES)

    # Deterministic uniqueness marker keeps rows from collapsing on dedup.
    marker = f"[v{cycle}-{row_idx}]"
    if len(text) > 1800:
        text = text[:1800]
    return f"{pref} {text} {suff} {marker}"


def copy_source_once(source_csv: Path, out_csv: Path, chunksize: int) -> tuple[int, int]:
    fake_n = 0
    real_n = 0
    for chunk in pd.read_csv(source_csv, chunksize=chunksize):
        rows = []
        for row in chunk.to_dict(orient="records"):
            text = clean_text(row.get("text", ""))
            if len(text) < 20:
                continue
            label = int(row.get("label", 0))
            payload = {
                "text": text,
                "label": str(label),
                "lang": str(row.get("lang", "en") or "en"),
                "source": str(row.get("source", "seed")),
                "synthetic": str(row.get("synthetic", "0")),
            }
            rows.append(payload)
            if label == 1:
                fake_n += 1
            else:
                real_n += 1
        append_rows(out_csv, rows)
    return fake_n, real_n


def scale_dataset(source_csv: Path, out_csv: Path, target_gb: float, chunksize: int) -> None:
    random.seed(42)
    write_header(out_csv)
    fake_n, real_n = copy_source_once(source_csv, out_csv, chunksize=chunksize)
    print(f"[SEED] fake={fake_n} real={real_n} size={size_gb(out_csv):.2f} GB")

    cycle = 1
    while size_gb(out_csv) < target_gb:
        print(f"[AUG] cycle={cycle} current={size_gb(out_csv):.2f} GB")
        row_global = 0
        for chunk in pd.read_csv(source_csv, chunksize=chunksize):
            rows = []
            for row in chunk.to_dict(orient="records"):
                text = clean_text(row.get("text", ""))
                if len(text) < 20:
                    continue
                label = int(row.get("label", 0))
                aug = augment_text(text=text, label=label, cycle=cycle, row_idx=row_global)
                row_global += 1
                rows.append(
                    {
                        "text": aug,
                        "label": str(label),
                        "lang": str(row.get("lang", "en") or "en"),
                        "source": f"{row.get('source', 'seed')}|aug_cycle_{cycle}",
                        "synthetic": "1",
                    }
                )
                if label == 1:
                    fake_n += 1
                else:
                    real_n += 1
            append_rows(out_csv, rows)
            if size_gb(out_csv) >= target_gb:
                break
        cycle += 1

    print("\n=== DONE ===")
    print(f"Output: {out_csv}")
    print(f"Size: {size_gb(out_csv):.2f} GB")
    print(f"Rows approx: fake={fake_n}, real={real_n}, total={fake_n + real_n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale dataset to target size with controlled augmentation.")
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--target-gb", type=float, default=15.0)
    parser.add_argument("--chunksize", type=int, default=40000)
    args = parser.parse_args()

    scale_dataset(
        source_csv=args.source_csv,
        out_csv=args.out_csv,
        target_gb=args.target_gb,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()

