"""
Trains the baseline recommender on raw click labels (Y).
Saves model to models/baseline.pt.

Run:
    python -m src.models.baseline
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TRAIN_DIR, BASELINE_MODEL, MODELS_DIR
from src.models.embeddings import RecommenderMLP, load_embeddings, build_dataset, MemmapDataset

EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3


def main():
    print("Loading embeddings...")
    emb = load_embeddings()

    print("Building dataset (click labels)...")
    dataset = build_dataset(TRAIN_DIR / "behaviors.tsv", emb, ytilde_map=None, tag="clicks")
    print(f"  {len(dataset):,} training examples")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RecommenderMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for user_emb, article_emb, label in loader:
            optimizer.zero_grad()
            pred = model(user_emb, article_emb)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{EPOCHS} — loss: {total_loss / len(loader):.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), BASELINE_MODEL)
    print(f"Saved to {BASELINE_MODEL}")


if __name__ == "__main__":
    main()
