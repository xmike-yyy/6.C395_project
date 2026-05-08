"""
Stage 3 evaluation for 6.C395_project.

Run from the repo root:

    python -m src.evaluate.metrics

Expected local inputs:
    data/dev/MINDsmall_dev/behaviors.tsv
    data/article_embeddings.npz
    data/article_scores.csv
    data/click_quality.csv          optional, but used if present
    data/y_tilde.csv                optional, but used if present
    models/baseline.pt
    models/informative.pt

Outputs:
    outputs/eval_metrics.csv
    outputs/topk_scores.csv
    outputs/disagreement_examples.csv
    outputs/informativeness_hist.png
    outputs/click_metrics_bar.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from src.config import (
    TRAIN_DIR,
    DEV_DIR,
    ARTICLE_SCORES_CSV,
    CLICK_QUALITY_CSV,
    Y_TILDE_CSV,
    BASELINE_MODEL,
    INFORMATIVE_MODEL,
)

from src.models.embeddings import (
    BEHAVIOR_COLS,
    EMBEDDING_DIM,
    NEWS_COLS,
    RecommenderMLP,
    load_embeddings,
)


OUT_DIR = Path("outputs")
KS = (5, 10)
# Threshold set to ~75th percentile of actual score distribution (~0.944).
# A fixed 0.5 threshold is useless: 99.5% of articles score above it.
INFO_THRESHOLD = 0.944
MAX_DISAGREEMENT_EXAMPLES = 100


@dataclass
class CandidateScores:
    news_ids: List[str]
    click_labels: np.ndarray
    baseline_scores: np.ndarray
    informative_scores: np.ndarray


def require_file(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise FileNotFoundError(msg)


def load_model(path: Path, device: torch.device) -> RecommenderMLP:
    require_file(path, "Make sure the trained checkpoints are present in models/.")

    model = RecommenderMLP().to(device)
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def load_models(device: torch.device) -> Dict[str, RecommenderMLP]:
    return {
        "baseline": load_model(BASELINE_MODEL, device),
        "informative": load_model(INFORMATIVE_MODEL, device),
    }


def load_behaviors() -> pd.DataFrame:
    behaviors_path = DEV_DIR / "behaviors.tsv"
    require_file(
        behaviors_path,
        "Download/extract MIND-small dev data, then run scripts/download_data.py if needed.",
    )

    df = pd.read_csv(
        behaviors_path,
        sep="\t",
        names=BEHAVIOR_COLS,
        dtype={
            "ImpressionID": str,
            "UserID": str,
            "Time": str,
            "History": str,
            "Impressions": str,
        },
    )

    df = df.dropna(subset=["Impressions"]).reset_index(drop=True)
    return df


def load_article_scores() -> Tuple[Dict[str, float], float]:
    require_file(
        ARTICLE_SCORES_CSV,
        "Expected data/article_scores.csv from Stage 1.",
    )

    df = pd.read_csv(ARTICLE_SCORES_CSV)

    if "NewsID" not in df.columns or "informativeness_score" not in df.columns:
        raise ValueError(
            f"{ARTICLE_SCORES_CSV} must have columns NewsID and informativeness_score. "
            f"Found: {list(df.columns)}"
        )

    scores = dict(zip(df["NewsID"], df["informativeness_score"].astype(float)))
    mean_score = float(df["informativeness_score"].mean())
    return scores, mean_score


def load_ytilde_scores() -> Dict[Tuple[str, str], float]:
    if not Y_TILDE_CSV.exists():
        print(f"Warning: {Y_TILDE_CSV} not found. y_tilde diagnostics will be empty.")
        return {}

    df = pd.read_csv(Y_TILDE_CSV)

    required = {"UserID", "NewsID", "y_tilde"}
    if not required.issubset(df.columns):
        print(
            f"Warning: {Y_TILDE_CSV} missing columns {required}. "
            f"Found: {list(df.columns)}. y_tilde diagnostics will be empty."
        )
        return {}

    return {
        (str(row.UserID), str(row.NewsID)): float(row.y_tilde)
        for row in df.itertuples(index=False)
    }


def load_click_quality_scores() -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    if not CLICK_QUALITY_CSV.exists():
        print(f"Warning: {CLICK_QUALITY_CSV} not found. click_quality diagnostics will be empty.")
        return {}, {}

    df = pd.read_csv(CLICK_QUALITY_CSV)

    if "quality" not in df.columns:
        print(
            f"Warning: {CLICK_QUALITY_CSV} missing quality column. "
            f"Found: {list(df.columns)}. click_quality diagnostics will be empty."
        )
        return {}, {}

    user_news_map: Dict[Tuple[str, str], float] = {}

    if {"UserID", "NewsID"}.issubset(df.columns):
        user_news_map = {
            (str(row.UserID), str(row.NewsID)): float(row.quality)
            for row in df.itertuples(index=False)
        }

    if "NewsID" not in df.columns:
        return user_news_map, {}

    news_map = df.groupby("NewsID")["quality"].mean().astype(float).to_dict()
    return user_news_map, news_map


def load_news_metadata() -> Dict[str, Dict[str, str]]:
    """
    Load article titles/categories when news.tsv is available.

    This is only used for disagreement_examples.csv.
    """
    metadata: Dict[str, Dict[str, str]] = {}

    for news_path in [TRAIN_DIR / "news.tsv", DEV_DIR / "news.tsv"]:
        if not news_path.exists():
            continue

        df = pd.read_csv(
            news_path,
            sep="\t",
            names=NEWS_COLS,
            dtype=str,
        ).fillna("")

        for row in df.itertuples(index=False):
            metadata[str(row.NewsID)] = {
                "title": str(row.Title),
                "category": str(row.Category),
                "subcategory": str(row.SubCategory),
                "abstract": str(row.Abstract),
            }

    return metadata


def parse_impressions(impressions: str) -> Tuple[List[str], np.ndarray]:
    news_ids: List[str] = []
    labels: List[int] = []

    for item in str(impressions).split():
        try:
            nid, label = item.rsplit("-", 1)
            news_ids.append(nid)
            labels.append(int(label))
        except ValueError:
            continue

    return news_ids, np.asarray(labels, dtype=np.int32)


def build_user_embedding(history: object, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    if not isinstance(history, str) or not history.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    vecs = [embeddings[nid] for nid in history.split() if nid in embeddings]

    if not vecs:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return np.mean(vecs, axis=0).astype(np.float32)


def score_candidates(
    model: RecommenderMLP,
    user_emb: np.ndarray,
    candidate_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[List[str], np.ndarray]:
    valid_ids = [nid for nid in candidate_ids if nid in embeddings]

    if not valid_ids:
        return [], np.asarray([], dtype=np.float32)

    article_batch = np.stack([embeddings[nid] for nid in valid_ids]).astype(np.float32)
    user_batch = np.repeat(user_emb[None, :], repeats=len(valid_ids), axis=0).astype(np.float32)

    article_tensor = torch.from_numpy(article_batch).to(device)
    user_tensor = torch.from_numpy(user_batch).to(device)

    with torch.inference_mode():
        scores = model(user_tensor, article_tensor).detach().cpu().numpy().astype(np.float32)

    return valid_ids, scores


def dcg_at_k(labels: np.ndarray, k: int) -> float:
    labels = np.asarray(labels, dtype=float)[:k]
    if labels.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(labels / discounts))


def ndcg_at_k(labels_in_rank_order: np.ndarray, k: int) -> float:
    actual = dcg_at_k(labels_in_rank_order, k)
    ideal = dcg_at_k(np.sort(labels_in_rank_order)[::-1], k)

    if ideal == 0.0:
        return 0.0

    return actual / ideal


def mrr(labels_in_rank_order: np.ndarray) -> float:
    hits = np.where(np.asarray(labels_in_rank_order) == 1)[0]

    if hits.size == 0:
        return 0.0

    return 1.0 / float(hits[0] + 1)


def precision_at_k(labels_in_rank_order: np.ndarray, k: int) -> float:
    labels = np.asarray(labels_in_rank_order)[:k]

    if labels.size == 0:
        return 0.0

    denom = min(k, labels.size)
    return float(labels.sum() / denom)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.size == 0 or len(np.unique(labels)) < 2:
        return np.nan

    return float(roc_auc_score(labels, scores))


def rank_indices(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores)[::-1]


def get_with_default(mapping: Dict[str, float], key: str, default: float) -> float:
    val = mapping.get(key, default)
    if pd.isna(val):
        return default
    return float(val)


def compute_rank_metrics_for_impression(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    order = rank_indices(scores)
    ranked_labels = labels[order]

    metrics = {
        "auc": safe_auc(labels, scores),
        "mrr": mrr(ranked_labels),
    }

    for k in KS:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_labels, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked_labels, k)

    return metrics


def average_metric_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}

    keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {}

    for key in keys:
        vals = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        out[key] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan

    return out


def add_topk_rows(
    rows: List[Dict[str, object]],
    model_name: str,
    impression_id: str,
    user_id: str,
    news_ids: List[str],
    click_labels: np.ndarray,
    scores: np.ndarray,
    article_score_map: Dict[str, float],
    article_score_mean: float,
    ytilde_map: Dict[Tuple[str, str], float],
    click_quality_user_news: Dict[Tuple[str, str], float],
    click_quality_news: Dict[str, float],
    news_metadata: Dict[str, Dict[str, str]],
) -> None:
    order = rank_indices(scores)

    for rank, idx in enumerate(order[: max(KS)], start=1):
        nid = news_ids[idx]
        article_score = get_with_default(article_score_map, nid, article_score_mean)

        y_key = (user_id, nid)
        y_tilde_present = y_key in ytilde_map
        y_tilde = float(ytilde_map[y_key]) if y_tilde_present else np.nan

        exact_quality_present = y_key in click_quality_user_news
        exact_quality = (
            float(click_quality_user_news[y_key]) if exact_quality_present else np.nan
        )

        news_quality_present = nid in click_quality_news
        news_quality = float(click_quality_news[nid]) if news_quality_present else np.nan
        news_bounce = 1.0 - news_quality if news_quality_present else np.nan

        meta = news_metadata.get(nid, {})

        rows.append(
            {
                "model": model_name,
                "impression_id": impression_id,
                "user_id": user_id,
                "rank": rank,
                "news_id": nid,
                "model_score": float(scores[idx]),
                "click_label": int(click_labels[idx]),
                "article_score": article_score,
                "info_hit_thr_0p5": int(article_score > INFO_THRESHOLD),
                "y_tilde": y_tilde,
                "y_tilde_present": int(y_tilde_present),
                "exact_click_quality": exact_quality,
                "exact_click_quality_present": int(exact_quality_present),
                "news_click_quality": news_quality,
                "news_click_quality_present": int(news_quality_present),
                "news_bounce": news_bounce,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "subcategory": meta.get("subcategory", ""),
            }
        )


def summarize_topk_metrics(topk_df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    model_df = topk_df[topk_df["model"] == model_name].copy()

    for k in KS:
        kdf = model_df[model_df["rank"] <= k]

        out[f"mean_article_score@{k}"] = float(kdf["article_score"].mean())
        out[f"info_precision@{k}_thr0.5"] = float(kdf["info_hit_thr_0p5"].mean())

        if "y_tilde_present" in kdf:
            present = kdf["y_tilde_present"].astype(bool)
            out[f"y_tilde_coverage@{k}"] = float(present.mean()) if len(kdf) else np.nan
            out[f"mean_y_tilde_observed@{k}"] = (
                float(kdf.loc[present, "y_tilde"].mean()) if present.any() else np.nan
            )

        if "news_click_quality_present" in kdf:
            present = kdf["news_click_quality_present"].astype(bool)
            out[f"news_quality_coverage@{k}"] = float(present.mean()) if len(kdf) else np.nan
            out[f"mean_news_click_quality_observed@{k}"] = (
                float(kdf.loc[present, "news_click_quality"].mean()) if present.any() else np.nan
            )
            out[f"mean_news_bounce_observed@{k}"] = (
                float(kdf.loc[present, "news_bounce"].mean()) if present.any() else np.nan
            )

    return out


def compute_disagreement_example(
    impression_id: str,
    user_id: str,
    news_ids: List[str],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    informative_scores: np.ndarray,
    article_score_map: Dict[str, float],
    article_score_mean: float,
    news_metadata: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, object]]:
    if len(news_ids) < 2:
        return None

    b_order = rank_indices(baseline_scores)
    i_order = rank_indices(informative_scores)

    b_top_idx = int(b_order[0])
    i_top_idx = int(i_order[0])

    if b_top_idx == i_top_idx:
        return None

    b_top_id = news_ids[b_top_idx]
    i_top_id = news_ids[i_top_idx]

    b_article_score = get_with_default(article_score_map, b_top_id, article_score_mean)
    i_article_score = get_with_default(article_score_map, i_top_id, article_score_mean)

    article_score_gain = i_article_score - b_article_score

    b_rank_of_i_top = int(np.where(b_order == i_top_idx)[0][0] + 1)
    i_rank_of_b_top = int(np.where(i_order == b_top_idx)[0][0] + 1)

    rank_flip_strength = (b_rank_of_i_top - 1) + (i_rank_of_b_top - 1)

    b_meta = news_metadata.get(b_top_id, {})
    i_meta = news_metadata.get(i_top_id, {})

    return {
        "impression_id": impression_id,
        "user_id": user_id,
        "baseline_top_news_id": b_top_id,
        "baseline_top_title": b_meta.get("title", ""),
        "baseline_top_category": b_meta.get("category", ""),
        "baseline_top_click_label": int(labels[b_top_idx]),
        "baseline_top_baseline_score": float(baseline_scores[b_top_idx]),
        "baseline_top_informative_score": float(informative_scores[b_top_idx]),
        "baseline_top_article_score": b_article_score,
        "baseline_top_rank_under_informative": i_rank_of_b_top,
        "informative_top_news_id": i_top_id,
        "informative_top_title": i_meta.get("title", ""),
        "informative_top_category": i_meta.get("category", ""),
        "informative_top_click_label": int(labels[i_top_idx]),
        "informative_top_baseline_score": float(baseline_scores[i_top_idx]),
        "informative_top_informative_score": float(informative_scores[i_top_idx]),
        "informative_top_article_score": i_article_score,
        "informative_top_rank_under_baseline": b_rank_of_i_top,
        "article_score_gain_informative_top_minus_baseline_top": article_score_gain,
        "rank_flip_strength": rank_flip_strength,
        "informative_picked_more_informative_article": int(article_score_gain > 0),
    }


def make_informativeness_hist(topk_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    for model_name in ["baseline", "informative"]:
        vals = topk_df[
            (topk_df["model"] == model_name) & (topk_df["rank"] <= 10)
        ]["article_score"].dropna()

        plt.hist(vals, bins=30, alpha=0.5, label=model_name)

    plt.xlabel("Article informativeness score")
    plt.ylabel("Top-10 recommendation count")
    plt.title("Top-10 informativeness distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "informativeness_hist.png", dpi=200)
    plt.close()


def make_click_metrics_bar(metrics_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    metrics = ["auc", "mrr", "ndcg@5", "ndcg@10", "precision@5", "precision@10"]
    available = [m for m in metrics if m in metrics_df.columns]

    if not available:
        return

    plot_df = metrics_df.set_index("model")[available].T

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title("Click-ranking metrics")
    ax.legend(title="Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "click_metrics_bar.png", dpi=200)
    plt.close()


def evaluate() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    models = load_models(device)

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print("Loading dev behaviors...")
    behaviors = load_behaviors()

    print("Loading auxiliary scores...")
    article_score_map, article_score_mean = load_article_scores()
    ytilde_map = load_ytilde_scores()
    click_quality_user_news, click_quality_news = load_click_quality_scores()
    news_metadata = load_news_metadata()

    print(f"Dev impressions: {len(behaviors):,}")
    print(f"Article embeddings: {len(embeddings):,}")
    print(f"Article informativeness scores: {len(article_score_map):,}")
    print(f"y_tilde entries: {len(ytilde_map):,}")
    print(f"News-level click-quality entries: {len(click_quality_news):,}")

    profile_cache: Dict[str, np.ndarray] = {}

    rank_metric_rows: Dict[str, List[Dict[str, float]]] = {
        "baseline": [],
        "informative": [],
    }

    topk_rows: List[Dict[str, object]] = []
    disagreement_rows: List[Dict[str, object]] = []

    total_candidate_count = 0
    valid_candidate_count = 0
    skipped_impressions = 0

    for row_num, row in enumerate(behaviors.itertuples(index=False), start=1):
        impression_id = str(row.ImpressionID)
        user_id = str(row.UserID)
        history = row.History if isinstance(row.History, str) else ""

        candidate_ids_raw, labels_raw = parse_impressions(row.Impressions)
        total_candidate_count += len(candidate_ids_raw)

        if not candidate_ids_raw:
            skipped_impressions += 1
            continue

        valid_mask = [nid in embeddings for nid in candidate_ids_raw]
        news_ids = [nid for nid, ok in zip(candidate_ids_raw, valid_mask) if ok]
        labels = labels_raw[np.asarray(valid_mask, dtype=bool)]

        valid_candidate_count += len(news_ids)

        if len(news_ids) < 2:
            skipped_impressions += 1
            continue

        if history not in profile_cache:
            profile_cache[history] = build_user_embedding(history, embeddings)

        user_emb = profile_cache[history]

        baseline_ids, baseline_scores = score_candidates(
            models["baseline"],
            user_emb,
            news_ids,
            embeddings,
            device,
        )

        informative_ids, informative_scores = score_candidates(
            models["informative"],
            user_emb,
            news_ids,
            embeddings,
            device,
        )

        if baseline_ids != news_ids or informative_ids != news_ids:
            raise RuntimeError("Internal scoring order mismatch.")

        scores_by_model = {
            "baseline": baseline_scores,
            "informative": informative_scores,
        }

        for model_name, scores in scores_by_model.items():
            rank_metric_rows[model_name].append(
                compute_rank_metrics_for_impression(labels, scores)
            )

            add_topk_rows(
                rows=topk_rows,
                model_name=model_name,
                impression_id=impression_id,
                user_id=user_id,
                news_ids=news_ids,
                click_labels=labels,
                scores=scores,
                article_score_map=article_score_map,
                article_score_mean=article_score_mean,
                ytilde_map=ytilde_map,
                click_quality_user_news=click_quality_user_news,
                click_quality_news=click_quality_news,
                news_metadata=news_metadata,
            )

        disagreement = compute_disagreement_example(
            impression_id=impression_id,
            user_id=user_id,
            news_ids=news_ids,
            labels=labels,
            baseline_scores=baseline_scores,
            informative_scores=informative_scores,
            article_score_map=article_score_map,
            article_score_mean=article_score_mean,
            news_metadata=news_metadata,
        )

        if disagreement is not None:
            disagreement_rows.append(disagreement)

        if row_num % 5000 == 0:
            print(f"Processed {row_num:,}/{len(behaviors):,} impressions...")

    print("Aggregating metrics...")

    topk_df = pd.DataFrame(topk_rows)

    metrics_rows: List[Dict[str, object]] = []

    for model_name in ["baseline", "informative"]:
        row: Dict[str, object] = {"model": model_name}
        row.update(average_metric_dicts(rank_metric_rows[model_name]))
        row.update(summarize_topk_metrics(topk_df, model_name))
        row["num_eval_impressions"] = len(rank_metric_rows[model_name])
        row["num_skipped_impressions"] = skipped_impressions
        row["total_candidate_count"] = total_candidate_count
        row["valid_candidate_count"] = valid_candidate_count
        row["candidate_embedding_coverage"] = (
            valid_candidate_count / total_candidate_count if total_candidate_count else np.nan
        )
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    disagreement_df = pd.DataFrame(disagreement_rows)

    if not disagreement_df.empty:
        disagreement_df = disagreement_df.sort_values(
            by=[
                "informative_picked_more_informative_article",
                "article_score_gain_informative_top_minus_baseline_top",
                "rank_flip_strength",
            ],
            ascending=[False, False, False],
        ).head(MAX_DISAGREEMENT_EXAMPLES)

    return metrics_df, topk_df, disagreement_df


def main() -> None:
    metrics_df, topk_df, disagreement_df = evaluate()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUT_DIR / "eval_metrics.csv"
    topk_path = OUT_DIR / "topk_scores.csv"
    disagreement_path = OUT_DIR / "disagreement_examples.csv"

    metrics_df.to_csv(metrics_path, index=False)
    topk_df.to_csv(topk_path, index=False)
    disagreement_df.to_csv(disagreement_path, index=False)

    make_informativeness_hist(topk_df)
    make_click_metrics_bar(metrics_df)

    print("\nSaved outputs:")
    print(f"  {metrics_path}")
    print(f"  {topk_path}")
    print(f"  {disagreement_path}")
    print(f"  {OUT_DIR / 'informativeness_hist.png'}")
    print(f"  {OUT_DIR / 'click_metrics_bar.png'}")

    print("\nEvaluation summary:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(metrics_df)


if __name__ == "__main__":
    main()