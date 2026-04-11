from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
FAKE_IN = DATA_DIR / "Fake.csv"
OUT = DATA_DIR / "all_fake.csv"

df = pd.read_csv(FAKE_IN)

# В ISOT обычно колонка text, но на всякий
text_col = "text" if "text" in df.columns else df.columns[0]

out = pd.DataFrame({
    "text": df[text_col].astype(str).fillna(""),
    "y": 1
})

out = out[out["text"].str.strip().str.len() > 0].drop_duplicates(subset=["text"])
out.to_csv(OUT, index=False, encoding="utf-8")

print("Saved:", OUT, "rows:", len(out))
