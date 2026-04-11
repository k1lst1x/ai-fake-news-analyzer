from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def clean_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: Path, max_rows: int, chunksize: int) -> pd.DataFrame:
    required = {"text", "label"}
    max_rows = int(max_rows if max_rows > 0 else 120_000)

    target_fake = max_rows // 2
    target_real = max_rows - target_fake
    picked_fake = 0
    picked_real = 0
    selected_parts: list[pd.DataFrame] = []

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
        raise RuntimeError("No rows selected for fine-tuning.")

    df = pd.concat(selected_parts, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    print(f"[LOAD] selected rows after dedup: {len(df)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune transformer fake-news classifier.")
    parser.add_argument("--data-csv", type=Path, default=Path("data/expanded/expanded_news_dataset.csv"))
    parser.add_argument("--base-model-dir", type=Path, default=Path("models/fake_news_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fake_news_model_ft"))
    parser.add_argument("--metrics-out", type=Path, default=Path("models/fine_tune_metrics.json"))
    parser.add_argument("--max-rows", type=int, default=120_000)
    parser.add_argument("--chunksize", type=int, default=120_000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    print(f"[INFO] Loading data from {args.data_csv}")
    df = load_data(args.data_csv, max_rows=args.max_rows, chunksize=args.chunksize)
    print(f"[INFO] Rows: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_dir)

    def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(
        tokenize, batched=True
    )
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    fp16 = torch.cuda.is_available()
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16 and not bf16,
        bf16=bf16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Fine-tuning started...")
    trainer.train()
    eval_metrics = trainer.evaluate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "metrics": {k: float(v) for k, v in eval_metrics.items()},
    }
    args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved fine-tuned model to {args.output_dir}")
    print(f"[INFO] Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
