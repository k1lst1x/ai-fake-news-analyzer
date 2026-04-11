from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "fake_news_model"
ALL_FAKE = DATA_DIR / "all_fake.csv"

MAX_TOKENS = 512
BATCH_SIZE = 16
STEP = 128

# ВАЖНО: из твоего train_eval получилось: LABEL_0_IS_FAKE
FAKE_LABEL = "LABEL_0"

df = pd.read_csv(ALL_FAKE)
texts = df["text"].astype(str).fillna("").tolist()

device = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "text-classification",
    model=str(MODEL_DIR),
    tokenizer=str(MODEL_DIR),
    device=device
)

pred_fake = 0
scores_fake = []

for i in tqdm(range(0, len(texts), STEP), desc="All-fake check", unit="chunk"):
    batch = texts[i:i+STEP]
    out = clf(batch, truncation=True, max_length=MAX_TOKENS, batch_size=BATCH_SIZE)

    for r in out:
        is_fake = (r["label"] == FAKE_LABEL)
        pred_fake += 1 if is_fake else 0

        # score тут "уверенность выбранного класса"
        if is_fake:
            scores_fake.append(float(r["score"]))

total = len(texts)
fake_rate = pred_fake / total if total else 0.0
avg_conf = (sum(scores_fake) / len(scores_fake)) if scores_fake else 0.0

print("\n===== ALL FAKE CHECK =====")
print("Total texts:", total)
print("Predicted fake:", pred_fake)
print("Predicted fake rate:", round(fake_rate, 4))
print("Avg confidence (only when predicted fake):", round(avg_conf, 4))
