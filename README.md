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

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

## What's Already in the Repo

Stages 1 and 2 are complete. The following outputs are committed and you don't need to regenerate them:

| File | Description |
|------|-------------|
| `data/article_scores.csv` | LLM informativeness scores for all 51K MIND articles |
| `data/click_quality.csv` | Behavioral depth-of-engagement signal (click quality) |
| `data/y_tilde.csv` | Final combined informativeness proxy label Ỹ |
| `models/baseline.pt` | Trained click-prediction model |
| `models/informative.pt` | Trained informativeness model |

**For Stage 3 (Ernesto):** clone the repo, run setup, download the MIND data (see below), then go straight to `python -m src.evaluate.metrics`.

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
```bash
python -m src.evaluate.metrics
```

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
  evaluate/             # Stage 3: compare models
    metrics.py
app/
  app.py                # Streamlit side-by-side demo
data/                   # gitignored
models/                 # gitignored
```

## Division of Work

| Stage | Owner |
|-------|-------|
| Data download | Michael |
| LLM scoring + behavioral signals (Stage 1) | Michael |
| Model training (Stage 2) | Michael |
| Evaluation (Stage 3) | Ernesto |
| Demo app | TBD |
