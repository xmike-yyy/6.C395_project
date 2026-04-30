# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIT 6.C395 final project. The core thesis: standard news recommenders optimize for clicks (Y), but we want to optimize for informativeness (Ỹ). Framed as a label bias problem — Y = Ỹ + Δ where Δ captures impulsive/outrage-driven clicks. We construct a proxy for Ỹ using Claude API scoring + behavioral signals from the MIND dataset, then train a separate recommendation model on that proxy and compare it to a click-trained baseline.

**Division of work:** Michael → Stages 0–2 (data, scoring, training). Ernesto → Stage 3 (evaluation).

## Repository Layout

```
scripts/
  download_data.py      # Stage 0: extract manually downloaded MIND-small zips into data/
src/
  scoring/              # Stage 1: construct Ỹ
    llm_judge.py        # HuggingFace zero-shot article scoring (no API cost)
    behavioral.py       # click-and-bounce signal extraction
    combine.py          # merge signals into final label
  models/               # Stage 2: train both recommenders
    embeddings.py       # shared article embedding utilities
    baseline.py         # trained on Y (clicks)
    informative.py      # trained on Ỹ
  evaluate/             # Stage 3: compare models
    metrics.py
app/
  app.py                # Streamlit side-by-side demo
data/                   # gitignored — populated by download script
models/                 # gitignored — saved model checkpoints
```

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with `ANTHROPIC_API_KEY=...`. Never commit it.

## Running the Pipeline

Each stage is run independently:

```bash
# Stage 0 — extract MIND-small into data/
# First: manually download MINDsmall_train.zip and MINDsmall_dev.zip from https://msnews.github.io/
# and place them in data/, then run:
python scripts/download_data.py

# Stage 1 — construct Ỹ
python scripts/download_model.py    # one-time: downloads bart-large-mnli → models/bart-mnli/
python -m src.scoring.llm_judge     # → data/article_scores.csv
python -m src.scoring.behavioral    # → data/click_quality.csv
python -m src.scoring.combine       # → data/y_tilde.csv

# Stage 2 — train models
python -m src.models.baseline       # → models/baseline.pt
python -m src.models.informative    # → models/informative.pt

# Stage 3 — evaluate
python -m src.evaluate.metrics

# Demo
streamlit run app/app.py
```

## MIND-Small Data Format

MIND-small has only `train` and `dev` splits (no test). The zips extract into nested folders — actual data lives at `data/train/MINDsmall_train/` and `data/dev/MINDsmall_dev/`. All paths are centralized in `src/config.py` (`TRAIN_DIR`, `DEV_DIR`). Each split contains:

- `news.tsv` — columns: `NewsID, Category, SubCategory, Title, Abstract, URL, TitleEntities, AbstractEntities`
- `behaviors.tsv` — columns: `ImpressionID, UserID, Time, History, Impressions`
  - `History`: space-separated NewsIDs the user previously clicked
  - `Impressions`: space-separated `NewsID-label` pairs (1=clicked, 0=not shown/clicked)

## Model Architecture

**Content-based embedding model** using `all-MiniLM-L6-v2` (sentence-transformers):
- Article embeddings precomputed once via `python -m src.models.embeddings` → `data/article_embeddings.npz`
- User profile = mean of clicked article embeddings from history
- MLP: `Linear(768→256) → ReLU → Dropout(0.2) → Linear(256→1) → Sigmoid`
- Input: `concat(user_emb, article_emb)` (384+384=768 dims)
- Trained with BCE loss, Adam optimizer, 5 epochs

**Critical invariant:** `RecommenderMLP` in `src/models/embeddings.py` is shared by both models. The only difference is the training label:
- `baseline.py` → `Y` (raw click: 0 or 1 from behaviors.tsv)
- `informative.py` → `Ỹ` (y_tilde from y_tilde.csv for clicked articles, 0 for not-clicked)

**Upgrade path: NRMS** (Neural News Recommendation with Multi-Head Self-Attention) if content-based results are too weak.

## Article Scoring (Stage 1)

`llm_judge.py` uses `facebook/bart-large-mnli` (zero-shot NLI) loaded from `models/bart-mnli/` (pre-downloaded via `scripts/download_model.py`). Labels: `["informative", "clickbait"]` — the probability assigned to "informative" is the score (0–1).

- **Pre-download required** — model lives in `models/bart-mnli/` (gitignored); run `download_model.py` once before scoring
- **Incremental disk caching** — appends to `data/article_scores.csv` after every batch; safe to kill and resume
- **Batch size 32** on CPU

## Behavioral Signal (Stage 1)

`behavioral.py` extracts a "depth-of-engagement" signal from `behaviors.tsv`:
- Within a session (same ImpressionID block or same-day clicks), if a user clicks article in category X and their *next* click is also category X → label that click as genuine interest (quality=1)
- Click followed by category jump → impulsive/shallow engagement (quality=0)
- Produces `data/click_quality.csv` with columns `UserID, NewsID, quality`

## Evaluation Recommendations (Stage 3 — Ernesto)

Suggested metrics for comparing baseline vs. informativeness model on held-out test impressions:

- **NDCG@K** — standard ranking metric; use `ranx` library
- **Precision@K on Ỹ** — fraction of each model's top-K that score above threshold on LLM informativeness
- **Click-and-bounce rate** — do baseline recommendations correlate more with the behavioral bounce signal?
- **Score distribution** — histogram of LLM informativeness scores for each model's top-K recommendations (the main visual for the paper)
- **AUC on held-out clicks** — sanity check: quantify how much click accuracy the informative model trades away

## Key Dependencies

```
anthropic
sentence-transformers
torch
pandas
numpy
scikit-learn
streamlit
python-dotenv
ranx
```
