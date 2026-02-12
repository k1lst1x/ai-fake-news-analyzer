from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR = (BASE_DIR / DATA_DIR).resolve()

FAKE_IN = DATA_DIR / "Fake.csv"
OUT = DATA_DIR / "all_fake.csv"

df = pd.read_csv(FAKE_IN)

text_col = "text" if "text" in df.columns else df.columns[0]

out = pd.DataFrame({
    "text": df[text_col].astype(str).fillna(""),
    "y": 1
})

out = out[out["text"].str.strip().str.len() > 0].drop_duplicates(subset=["text"])
out.to_csv(OUT, index=False, encoding="utf-8")

print("Saved:", OUT, "rows:", len(out))
