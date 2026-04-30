"""
Scores each article for informativeness using facebook/bart-large-mnli (zero-shot NLI).

Requires the model to be downloaded locally first:
    python scripts/download_model.py

Outputs data/article_scores.csv with columns: NewsID, informativeness_score (0-1).
Safe to re-run — resumes from where it left off.
"""

import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline

from src.config import TRAIN_DIR, ARTICLE_SCORES_CSV

NEWS_TSV = TRAIN_DIR / "news.tsv"
OUTPUT_CSV = ARTICLE_SCORES_CSV
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "bart-mnli"

COLS = ["NewsID", "Category", "SubCategory", "Title", "Abstract",
        "URL", "TitleEntities", "AbstractEntities"]
LABELS = ["informative", "clickbait"]
BATCH_SIZE = 32
CHUNK_SIZE = 3000


def load_scored_ids() -> set:
    if OUTPUT_CSV.exists():
        return set(pd.read_csv(OUTPUT_CSV)["NewsID"].tolist())
    return set()


def main():
    if not MODEL_DIR.exists():
        print("Model not found. Run: python scripts/download_model.py")
        return

    df = pd.read_csv(NEWS_TSV, sep="\t", names=COLS)
    df["Abstract"] = df["Abstract"].fillna("")

    scored_ids = load_scored_ids()
    remaining = df[~df["NewsID"].isin(scored_ids)].reset_index(drop=True)
    print(f"Total: {len(df)} | Scored: {len(scored_ids)} | Remaining: {len(remaining)}")

    if remaining.empty:
        print("All articles already scored.")
        return

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model from {MODEL_DIR}...")
    classifier = pipeline(
        "zero-shot-classification",
        model=str(MODEL_DIR),
        device=device,
    )

    chunk = remaining.head(CHUNK_SIZE).reset_index(drop=True)
    print(f"Scoring next {len(chunk)} articles (then stop and re-run to continue)...")

    texts = (chunk["Title"] + ". " + chunk["Abstract"]).tolist()
    news_ids = chunk["NewsID"].tolist()
    write_header = not OUTPUT_CSV.exists()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_ids = news_ids[i : i + BATCH_SIZE]

        outputs = classifier(batch_texts, LABELS)

        rows = []
        for news_id, out in zip(batch_ids, outputs):
            score = out["scores"][out["labels"].index("informative")]
            rows.append({"NewsID": news_id, "informativeness_score": round(score, 4)})

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
        write_header = False

        done = min(i + BATCH_SIZE, len(texts))
        print(f"  {done}/{len(texts)} scored this run", end="\r")

    total_scored = len(scored_ids) + len(chunk)
    print(f"\nDone. {total_scored}/{len(df)} total scored. Re-run to continue.")


if __name__ == "__main__":
    main()
