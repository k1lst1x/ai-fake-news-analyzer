import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = (BASE_DIR / os.getenv("MODEL_DIR", "./models/fake_news_model")).resolve()
DATA_DIR = (BASE_DIR / os.getenv("DATA_DIR", "./data")).resolve()

TRUE = DATA_DIR / "True.csv"
FAKE = DATA_DIR / "Fake.csv"

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
mdl.to(device).eval()

def pred(text):
    inp = tok(text, truncation=True, max_length=512, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        p = F.softmax(mdl(**inp).logits, dim=-1).squeeze(0).cpu().tolist()
    return p

true_df = pd.read_csv(TRUE)
fake_df = pd.read_csv(FAKE)

t = str(true_df["text"].dropna().iloc[0])
f = str(fake_df["text"].dropna().iloc[0])

print("TRUE sample probs [class0, class1]:", pred(t))
print("FAKE sample probs [class0, class1]:", pred(f))
