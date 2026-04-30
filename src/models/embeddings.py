"""
Shared utilities: article embedding precomputation and RecommenderMLP architecture.

Run standalone to precompute and cache article embeddings:
    python -m src.models.embeddings
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

from src.config import TRAIN_DIR, EMBEDDINGS_NPZ, DATA_DIR

NEWS_TSV = TRAIN_DIR / "news.tsv"
NEWS_COLS = ["NewsID", "Category", "SubCategory", "Title", "Abstract",
             "URL", "TitleEntities", "AbstractEntities"]
BEHAVIOR_COLS = ["ImpressionID", "UserID", "Time", "History", "Impressions"]

EMBEDDING_DIM = 384
SENTENCE_MODEL = "all-MiniLM-L6-v2"


class RecommenderMLP(nn.Module):
    def __init__(self, dim: int = EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_emb: torch.Tensor, article_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_emb, article_emb], dim=-1)
        return self.net(x).squeeze(-1)


class MemmapDataset(Dataset):
    """Reads training examples from memory-mapped files — no full RAM load."""

    def __init__(self, user_path, article_path, labels_path):
        meta = json.loads(Path(str(user_path) + ".meta").read_text())
        n, dim = meta["n"], meta["dim"]
        self.user = np.memmap(user_path, dtype="float32", mode="r", shape=(n, dim))
        self.article = np.memmap(article_path, dtype="float32", mode="r", shape=(n, dim))
        self.labels = np.memmap(labels_path, dtype="float32", mode="r", shape=(n,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.user[idx].copy()),
            torch.from_numpy(self.article[idx].copy()),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )


def compute_and_save():
    print("Loading articles...")
    df = pd.read_csv(NEWS_TSV, sep="\t", names=NEWS_COLS)
    df["Abstract"] = df["Abstract"].fillna("")
    texts = (df["Title"] + ". " + df["Abstract"]).tolist()
    ids = df["NewsID"].tolist()

    print(f"Encoding {len(texts)} articles with {SENTENCE_MODEL}...")
    model = SentenceTransformer(SENTENCE_MODEL)
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=True)

    EMBEDDINGS_NPZ.parent.mkdir(exist_ok=True)
    np.savez(EMBEDDINGS_NPZ, ids=np.array(ids), embeddings=embeddings.astype(np.float32))
    print(f"Saved {embeddings.shape} embeddings to {EMBEDDINGS_NPZ}")


def load_embeddings():
    if not EMBEDDINGS_NPZ.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_NPZ} not found. Run: python -m src.models.embeddings"
        )
    data = np.load(EMBEDDINGS_NPZ)
    return dict(zip(data["ids"].tolist(), data["embeddings"]))


def build_dataset(behaviors_path, emb: dict, ytilde_map=None, tag="clicks"):
    """
    Streams (user_emb, article_emb, label) to memory-mapped files on disk.
    Returns a MemmapDataset — only reads from disk during training, not into RAM.

    tag: filename tag to distinguish baseline ("clicks") from informative ("ytilde").
    """
    user_path    = DATA_DIR / f"ds_user_{tag}.dat"
    article_path = DATA_DIR / f"ds_article_{tag}.dat"
    labels_path  = DATA_DIR / f"ds_labels_{tag}.dat"

    # Return cached dataset if already built
    if user_path.exists() and Path(str(user_path) + ".meta").exists():
        print(f"  Loading cached dataset ({tag}) from disk...")
        return MemmapDataset(user_path, article_path, labels_path)

    behaviors = pd.read_csv(behaviors_path, sep="\t", names=BEHAVIOR_COLS)
    behaviors = behaviors.dropna(subset=["Impressions"]).reset_index(drop=True)

    # Precompute unique user profiles (one per unique history string)
    print("  Computing user profiles...")
    profile_map = {}
    for hist_str in behaviors["History"].fillna("").unique():
        nids = hist_str.split() if hist_str else []
        vecs = [emb[n] for n in nids if n in emb]
        profile_map[hist_str] = (
            np.mean(vecs, axis=0).astype(np.float32) if vecs
            else np.zeros(EMBEDDING_DIM, dtype=np.float32)
        )

    # Count total valid examples (need size upfront for memmap)
    print("  Counting examples...")
    total = sum(
        1
        for imp_str in behaviors["Impressions"]
        for item in imp_str.split()
        if item.rsplit("-", 1)[0] in emb
    )
    print(f"  {total:,} examples — streaming to disk...")

    user_mm    = np.memmap(user_path,    dtype="float32", mode="w+", shape=(total, EMBEDDING_DIM))
    article_mm = np.memmap(article_path, dtype="float32", mode="w+", shape=(total, EMBEDDING_DIM))
    labels_mm  = np.memmap(labels_path,  dtype="float32", mode="w+", shape=(total,))

    idx = 0
    for i, row in behaviors.iterrows():
        hist_str = row["History"] if pd.notna(row["History"]) else ""
        user_emb = profile_map[hist_str]
        user_id  = row["UserID"]

        for item in row["Impressions"].split():
            nid, click_str = item.rsplit("-", 1)
            clicked = int(click_str)
            if nid not in emb:
                continue

            label = (
                ytilde_map.get((user_id, nid), 0.0) if (ytilde_map and clicked) else
                0.0 if ytilde_map else
                float(clicked)
            )

            user_mm[idx]    = user_emb
            article_mm[idx] = emb[nid]
            labels_mm[idx]  = label
            idx += 1

        if i % 10000 == 0:
            user_mm.flush(); article_mm.flush(); labels_mm.flush()
            print(f"  {i}/{len(behaviors)} impressions processed", end="\r")

    user_mm.flush(); article_mm.flush(); labels_mm.flush()

    # Save metadata so MemmapDataset can reload without knowing shape
    meta = json.dumps({"n": total, "dim": EMBEDDING_DIM})
    Path(str(user_path) + ".meta").write_text(meta)

    print(f"\n  Done — {total:,} examples saved to disk.")
    return MemmapDataset(user_path, article_path, labels_path)


if __name__ == "__main__":
    compute_and_save()
