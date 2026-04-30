# Informative News Recommender

MIT 6.C395 Final Project — Ernesto Gomez & Michael Sun

## What This Is

Standard news recommenders optimize for clicks. We build one that optimizes for **informativeness** instead. The core framing is a label bias problem: the click label Y = Ỹ + Δ, where Ỹ is whether a user was genuinely informed and Δ captures impulsive/outrage-driven behavior. We construct a proxy for Ỹ using LLM scoring + behavioral signals from the MIND dataset, train a recommendation model on it, and compare it to a standard click-trained baseline.

## Setup

**Requirements:** Python 3.9+

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** A `.env` file with `ANTHROPIC_API_KEY=...` is only needed if re-running Stage 1 scoring. Stage 3 evaluation does not use the Anthropic API.

## What's Already in the Repo

Stages 1 and 2 are complete. The following outputs are committed and you don't need to regenerate them:

| File | Description |
|------|-------------|
| `data/article_scores.csv` | LLM informativeness scores for all 51K MIND articles |
| `data/click_quality.csv` | Behavioral depth-of-engagement signal (click quality) |
| `data/y_tilde.csv` | Final combined informativeness proxy label Ỹ |
| `models/baseline.pt` | Trained click-prediction model |
| `models/informative.pt` | Trained informativeness model |

**For Stage 3 (Ernesto):** see the [Stage 3 Handoff](#stage-3-handoff-ernesto) section below.

## Getting the Data

Microsoft restricts direct downloads of MIND-small. You need to grab the zips manually first:

1. Go to **https://msnews.github.io/** and download `MINDsmall_train.zip` and `MINDsmall_dev.zip`
2. Place both zips in the `data/` directory
3. Run the extraction script:

```bash
python scripts/download_data.py
```

This extracts the zips into `data/train/` and `data/dev/` and is safe to re-run.

Each split contains:
- `news.tsv` — `NewsID, Category, SubCategory, Title, Abstract, URL, TitleEntities, AbstractEntities`
- `behaviors.tsv` — `ImpressionID, UserID, Time, History, Impressions`

## Pipeline

The project runs in four independent stages:

### Stage 1 — Construct Informativeness Label (Ỹ)
```bash
python scripts/download_model.py    # one-time: download bart-large-mnli (~1.6GB) → models/bart-mnli/
python -m src.scoring.llm_judge     # zero-shot scoring → data/article_scores.csv
python -m src.scoring.behavioral    # click-and-bounce signals → data/click_quality.csv
python -m src.scoring.combine       # merge into final label  → data/y_tilde.csv
```

### Stage 2 — Train Models
```bash
python -m src.models.baseline       # click-prediction model  → models/baseline.pt
python -m src.models.informative    # informativeness model   → models/informative.pt
```

### Stage 3 — Evaluate

See the [Stage 3 Handoff](#stage-3-handoff-ernesto) section.

### Demo
```bash
streamlit run app/app.py
```

## Project Structure

```
scripts/
  download_data.py      # data download utility
src/
  scoring/              # Stage 1: build Ỹ
    llm_judge.py
    behavioral.py
    combine.py
  models/               # Stage 2: train recommenders
    embeddings.py
    baseline.py
    informative.py
  evaluate/             # Stage 3: compare models (Ernesto)
    metrics.py          # ← needs to be written
data/
  article_scores.csv    # committed — Stage 1 output
  click_quality.csv     # committed — Stage 1 output
  y_tilde.csv           # committed — Stage 1 output
  train/, dev/          # not committed — download from msnews.github.io
  article_embeddings.npz  # not committed — regenerate via embeddings.py
models/
  baseline.pt           # committed — trained model
  informative.pt        # committed — trained model
  bart-mnli/            # not committed — download via scripts/download_model.py
```

## Division of Work

| Stage | Owner |
|-------|-------|
| Data download | Michael |
| LLM scoring + behavioral signals (Stage 1) | Michael |
| Model training (Stage 2) | Michael |
| Evaluation (Stage 3) | Ernesto |
| Demo app | TBD |

---

## Stage 3 Handoff (Ernesto)

Stages 1 and 2 are done. The trained models and all label CSVs are in the repo. Here's exactly what you need to do.

### Step 1 — Clone and install

```bash
git clone https://github.com/xmike-yyy/6.C395_project.git
cd 6.C395_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Download MIND-small

1. Go to **https://msnews.github.io/** and download `MINDsmall_train.zip` and `MINDsmall_dev.zip`
2. Place both zips in `data/`
3. Extract:

```bash
python scripts/download_data.py
```

### Step 3 — Regenerate article embeddings

The embeddings file is too large to commit (76MB). Regenerate it once (~5 min on CPU):

```bash
python -m src.models.embeddings
```

This writes `data/article_embeddings.npz` and is required before running evaluation.

### Step 4 — Write `src/evaluate/metrics.py`

This file doesn't exist yet — it's your task. Load both trained models and run them on the dev split (`data/dev/MINDsmall_dev/behaviors.tsv`). Suggested metrics:

- **NDCG@K** — standard ranking metric; use the `ranx` library
- **Precision@K on Ỹ** — fraction of each model's top-K with `y_tilde > threshold` (threshold ≈ 0.5); load scores from `data/article_scores.csv`
- **Click-and-bounce rate** — do baseline recommendations correlate more with the behavioral bounce signal in `data/click_quality.csv`?
- **Score distribution** — histogram of LLM informativeness scores for each model's top-K (the main visual for the paper)
- **AUC on held-out clicks** — sanity check: how much click accuracy does the informative model trade away?

To load a model:

```python
import torch
from src.models.embeddings import RecommenderMLP
from src.config import BASELINE_MODEL, INFORMATIVE_MODEL

baseline = RecommenderMLP()
baseline.load_state_dict(torch.load(BASELINE_MODEL, map_location="cpu"))
baseline.eval()

informative = RecommenderMLP()
informative.load_state_dict(torch.load(INFORMATIVE_MODEL, map_location="cpu"))
informative.eval()
```

User embeddings are built the same way training built them — mean of clicked article embeddings from the history field. See `src/models/embeddings.py` (`build_dataset`) for the exact logic.
