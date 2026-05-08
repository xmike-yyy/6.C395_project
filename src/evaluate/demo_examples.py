from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
    NEWS_COLS,
    EMBEDDING_DIM,
    RecommenderMLP,
    load_embeddings,
)


OUT_DIR = Path("outputs/demo_examples")


def require_file(path: Path, hint: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\n{hint}"
        raise FileNotFoundError(msg)


def load_model(path: Path) -> RecommenderMLP:
    require_file(path, "Expected trained model checkpoint in models/.")
    model = RecommenderMLP()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_behaviors() -> pd.DataFrame:
    path = DEV_DIR / "behaviors.tsv"
    require_file(
        path,
        "Download MINDsmall_dev.zip, put it in data/, then run python scripts/download_data.py.",
    )
    return pd.read_csv(path, sep="\t", names=BEHAVIOR_COLS, dtype=str).dropna(
        subset=["Impressions"]
    )


def load_news_metadata() -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}

    for path in [TRAIN_DIR / "news.tsv", DEV_DIR / "news.tsv"]:
        if not path.exists():
            continue

        df = pd.read_csv(path, sep="\t", names=NEWS_COLS, dtype=str).fillna("")

        for row in df.itertuples(index=False):
            metadata[str(row.NewsID)] = {
                "title": str(row.Title),
                "abstract": str(row.Abstract),
                "category": str(row.Category),
                "subcategory": str(row.SubCategory),
                "url": str(row.URL),
            }

    return metadata


def load_article_scores() -> Dict[str, float]:
    require_file(ARTICLE_SCORES_CSV, "Expected Stage 1 output: data/article_scores.csv.")
    df = pd.read_csv(ARTICLE_SCORES_CSV)

    if "NewsID" not in df.columns:
        raise ValueError(f"{ARTICLE_SCORES_CSV} needs a NewsID column. Found {df.columns.tolist()}")

    possible_score_cols = [
        "informativeness_score",
        "article_score",
        "score",
        "llm_score",
    ]

    score_col = next((c for c in possible_score_cols if c in df.columns), None)

    if score_col is None:
        numeric_cols = [
            c for c in df.columns
            if c != "NewsID" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise ValueError(
                f"Could not infer article informativeness score column from {df.columns.tolist()}"
            )
        score_col = numeric_cols[0]

    return dict(zip(df["NewsID"].astype(str), df[score_col].astype(float)))


def load_y_tilde() -> Dict[Tuple[str, str], float]:
    if not Y_TILDE_CSV.exists():
        return {}

    df = pd.read_csv(Y_TILDE_CSV)

    if not {"UserID", "NewsID"}.issubset(df.columns):
        return {}

    possible_cols = ["y_tilde", "Y_tilde", "label", "informativeness_label"]
    y_col = next((c for c in possible_cols if c in df.columns), None)

    if y_col is None:
        numeric_cols = [
            c for c in df.columns
            if c not in {"UserID", "NewsID"} and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return {}
        y_col = numeric_cols[0]

    return {
        (str(row.UserID), str(row.NewsID)): float(getattr(row, y_col))
        for row in df.itertuples(index=False)
    }


def load_click_quality() -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    if not CLICK_QUALITY_CSV.exists():
        return {}, {}

    df = pd.read_csv(CLICK_QUALITY_CSV)

    if "quality" not in df.columns or "NewsID" not in df.columns:
        return {}, {}

    user_article_quality = {}
    if "UserID" in df.columns:
        user_article_quality = {
            (str(row.UserID), str(row.NewsID)): float(row.quality)
            for row in df.itertuples(index=False)
        }

    article_quality = df.groupby("NewsID")["quality"].mean().astype(float).to_dict()
    article_quality = {str(k): float(v) for k, v in article_quality.items()}

    return user_article_quality, article_quality


def parse_impressions(impressions: str) -> Tuple[List[str], np.ndarray]:
    news_ids = []
    click_labels = []

    for item in str(impressions).split():
        try:
            news_id, click = item.rsplit("-", 1)
            news_ids.append(news_id)
            click_labels.append(int(click))
        except ValueError:
            continue

    return news_ids, np.asarray(click_labels, dtype=np.float32)


def build_user_embedding(history: object, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    if not isinstance(history, str) or not history.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    vecs = [embeddings[nid] for nid in history.split() if nid in embeddings]

    if not vecs:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return np.mean(vecs, axis=0).astype(np.float32)


def score_candidates(
    model: RecommenderMLP,
    user_embedding: np.ndarray,
    news_ids: List[str],
    embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    article_embeddings = np.stack([embeddings[nid] for nid in news_ids]).astype(np.float32)
    user_batch = np.repeat(user_embedding[None, :], len(news_ids), axis=0).astype(np.float32)

    with torch.inference_mode():
        scores = model(
            torch.from_numpy(user_batch),
            torch.from_numpy(article_embeddings),
        ).numpy()

    return scores.astype(float)


def add_article_context(
    row: dict,
    news_id: str,
    user_id: str,
    metadata: Dict[str, Dict[str, str]],
    article_scores: Dict[str, float],
    y_tilde: Dict[Tuple[str, str], float],
    user_article_quality: Dict[Tuple[str, str], float],
    article_quality: Dict[str, float],
) -> dict:
    meta = metadata.get(news_id, {})

    row.update(
        {
            "news_id": news_id,
            "title": meta.get("title", ""),
            "abstract": meta.get("abstract", ""),
            "category": meta.get("category", ""),
            "subcategory": meta.get("subcategory", ""),
            "url": meta.get("url", ""),
            "article_informativeness_score": article_scores.get(news_id, np.nan),
            "y_tilde": y_tilde.get((user_id, news_id), np.nan),
            "user_article_click_quality": user_article_quality.get((user_id, news_id), np.nan),
            "article_avg_click_quality": article_quality.get(news_id, np.nan),
        }
    )

    return row


def make_demo_rows(
    max_impressions: int,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading models...")
    baseline = load_model(BASELINE_MODEL)
    informative = load_model(INFORMATIVE_MODEL)

    print("Loading data...")
    embeddings = load_embeddings()
    behaviors = load_behaviors()
    metadata = load_news_metadata()
    article_scores = load_article_scores()
    y_tilde = load_y_tilde()
    user_article_quality, article_quality = load_click_quality()

    if max_impressions:
        behaviors = behaviors.head(max_impressions)

    model_rows = []
    disagreement_rows = []
    label_disagreement_rows = []

    profile_cache: Dict[str, np.ndarray] = {}

    print(f"Scoring {len(behaviors):,} dev impressions...")

    for idx, behavior in enumerate(behaviors.itertuples(index=False), start=1):
        impression_id = str(behavior.ImpressionID)
        user_id = str(behavior.UserID)
        history = behavior.History if isinstance(behavior.History, str) else ""

        raw_news_ids, raw_clicks = parse_impressions(behavior.Impressions)

        valid_pairs = [
            (nid, click)
            for nid, click in zip(raw_news_ids, raw_clicks)
            if nid in embeddings
        ]

        if len(valid_pairs) < 2:
            continue

        news_ids = [nid for nid, _ in valid_pairs]
        clicks = np.asarray([click for _, click in valid_pairs], dtype=float)

        if history not in profile_cache:
            profile_cache[history] = build_user_embedding(history, embeddings)

        user_embedding = profile_cache[history]

        baseline_scores = score_candidates(baseline, user_embedding, news_ids, embeddings)
        informative_scores = score_candidates(informative, user_embedding, news_ids, embeddings)

        score_bundle = {
            "baseline": baseline_scores,
            "informative": informative_scores,
        }

        for model_name, scores in score_bundle.items():
            top_order = np.argsort(scores)[::-1][:top_n]
            bottom_order = np.argsort(scores)[:top_n]

            for group, order in [("top", top_order), ("bottom", bottom_order)]:
                for rank, candidate_idx in enumerate(order, start=1):
                    nid = news_ids[candidate_idx]

                    row = {
                        "model": model_name,
                        "group": group,
                        "rank_within_group": rank,
                        "impression_id": impression_id,
                        "user_id": user_id,
                        "click_label": int(clicks[candidate_idx]),
                        "baseline_model_score": float(baseline_scores[candidate_idx]),
                        "informative_model_score": float(informative_scores[candidate_idx]),
                    }

                    model_rows.append(
                        add_article_context(
                            row,
                            nid,
                            user_id,
                            metadata,
                            article_scores,
                            y_tilde,
                            user_article_quality,
                            article_quality,
                        )
                    )

        baseline_top_idx = int(np.argmax(baseline_scores))
        informative_top_idx = int(np.argmax(informative_scores))

        if baseline_top_idx != informative_top_idx:
            b_id = news_ids[baseline_top_idx]
            i_id = news_ids[informative_top_idx]

            b_rank_under_info = int(
                np.where(np.argsort(informative_scores)[::-1] == baseline_top_idx)[0][0] + 1
            )
            i_rank_under_base = int(
                np.where(np.argsort(baseline_scores)[::-1] == informative_top_idx)[0][0] + 1
            )

            b_article_info = article_scores.get(b_id, np.nan)
            i_article_info = article_scores.get(i_id, np.nan)

            row = {
                "impression_id": impression_id,
                "user_id": user_id,
                "baseline_top_news_id": b_id,
                "informative_top_news_id": i_id,
                "baseline_top_title": metadata.get(b_id, {}).get("title", ""),
                "informative_top_title": metadata.get(i_id, {}).get("title", ""),
                "baseline_top_click_label": int(clicks[baseline_top_idx]),
                "informative_top_click_label": int(clicks[informative_top_idx]),
                "baseline_top_baseline_score": float(baseline_scores[baseline_top_idx]),
                "baseline_top_informative_score": float(informative_scores[baseline_top_idx]),
                "informative_top_baseline_score": float(baseline_scores[informative_top_idx]),
                "informative_top_informative_score": float(informative_scores[informative_top_idx]),
                "baseline_top_article_informativeness": b_article_info,
                "informative_top_article_informativeness": i_article_info,
                "informativeness_gain_info_minus_baseline": i_article_info - b_article_info,
                "baseline_top_rank_under_informative_model": b_rank_under_info,
                "informative_top_rank_under_baseline_model": i_rank_under_base,
                "rank_disagreement_strength": b_rank_under_info + i_rank_under_base,
            }

            disagreement_rows.append(row)

        for candidate_idx, nid in enumerate(news_ids):
            article_info = article_scores.get(nid, np.nan)
            yt = y_tilde.get((user_id, nid), np.nan)
            click = clicks[candidate_idx]

            article_label_gap = abs(click - article_info) if not np.isnan(article_info) else np.nan
            ytilde_label_gap = abs(click - yt) if not np.isnan(yt) else np.nan

            if (
                (not np.isnan(article_label_gap) and article_label_gap >= 0.75)
                or (not np.isnan(ytilde_label_gap) and ytilde_label_gap >= 0.75)
            ):
                row = {
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "click_label": int(click),
                    "baseline_model_score": float(baseline_scores[candidate_idx]),
                    "informative_model_score": float(informative_scores[candidate_idx]),
                    "article_label_gap_abs_click_minus_info": article_label_gap,
                    "ytilde_label_gap_abs_click_minus_ytilde": ytilde_label_gap,
                }

                label_disagreement_rows.append(
                    add_article_context(
                        row,
                        nid,
                        user_id,
                        metadata,
                        article_scores,
                        y_tilde,
                        user_article_quality,
                        article_quality,
                    )
                )

        if idx % 1000 == 0:
            print(f"Processed {idx:,}/{len(behaviors):,} impressions...")

    model_df = pd.DataFrame(model_rows)
    disagreement_df = pd.DataFrame(disagreement_rows)
    label_disagreement_df = pd.DataFrame(label_disagreement_rows)

    if not disagreement_df.empty:
        disagreement_df = disagreement_df.sort_values(
            ["informativeness_gain_info_minus_baseline", "rank_disagreement_strength"],
            ascending=[False, False],
        )

    if not label_disagreement_df.empty:
        label_disagreement_df = label_disagreement_df.sort_values(
            [
                "article_label_gap_abs_click_minus_info",
                "ytilde_label_gap_abs_click_minus_ytilde",
            ],
            ascending=[False, False],
            na_position="last",
        )

    return model_df, disagreement_df, label_disagreement_df


def write_markdown_summary(
    model_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
    label_disagreement_df: pd.DataFrame,
) -> None:
    path = OUT_DIR / "demo_examples.md"

    lines = []
    lines.append("# Demo Examples")
    lines.append("")
    lines.append("This file gives qualitative examples for the baseline and informative recommenders.")
    lines.append("The baseline model is trained on ordinary click labels. The informative model is trained on the project's informativeness proxy.")
    lines.append("")

    lines.append("## Example top recommendations by model")
    lines.append("")

    for model_name in ["baseline", "informative"]:
        lines.append(f"### {model_name}")
        sample = model_df[
            (model_df["model"] == model_name)
            & (model_df["group"] == "top")
        ].head(10)

        for row in sample.itertuples(index=False):
            lines.append(
                f"- Rank {row.rank_within_group}: {row.title} "
                f"(click={row.click_label}, article_info={row.article_informativeness_score:.3f}, "
                f"baseline_score={row.baseline_model_score:.3f}, "
                f"informative_score={row.informative_model_score:.3f})"
            )

        lines.append("")

    lines.append("## Strong model disagreements")
    lines.append("")

    for row in disagreement_df.head(10).itertuples(index=False):
        lines.append(f"### Impression {row.impression_id}")
        lines.append(
            f"- Baseline top: {row.baseline_top_title} "
            f"(article_info={row.baseline_top_article_informativeness:.3f}, "
            f"click={row.baseline_top_click_label})"
        )
        lines.append(
            f"- Informative top: {row.informative_top_title} "
            f"(article_info={row.informative_top_article_informativeness:.3f}, "
            f"click={row.informative_top_click_label})"
        )
        lines.append(
            f"- Informativeness gain: {row.informativeness_gain_info_minus_baseline:.3f}"
        )
        lines.append("")

    lines.append("## Articles with very different click and informativeness labels")
    lines.append("")

    for row in label_disagreement_df.head(15).itertuples(index=False):
        lines.append(
            f"- {row.title} "
            f"(click={row.click_label}, article_info={row.article_informativeness_score:.3f}, "
            f"y_tilde={row.y_tilde if not pd.isna(row.y_tilde) else 'NA'}, "
            f"baseline_score={row.baseline_model_score:.3f}, "
            f"informative_score={row.informative_model_score:.3f})"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(model_df: pd.DataFrame, disagreement_df: pd.DataFrame) -> None:
    if model_df.empty:
        return

    top_df = model_df[model_df["group"] == "top"].copy()

    plt.figure(figsize=(8, 5))
    top_df.boxplot(
        column="article_informativeness_score",
        by="model",
        grid=False,
    )
    plt.suptitle("")
    plt.title("Informativeness scores among top recommendations")
    plt.xlabel("Model")
    plt.ylabel("Article informativeness score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_recommendation_informativeness_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        model_df["baseline_model_score"],
        model_df["informative_model_score"],
        alpha=0.35,
        s=10,
    )
    plt.xlabel("Baseline model score")
    plt.ylabel("Informative model score")
    plt.title("Candidate scores under both models")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_score_scatter.png", dpi=200)
    plt.close()

    if not disagreement_df.empty:
        plt.figure(figsize=(8, 5))
        disagreement_df.head(50)["informativeness_gain_info_minus_baseline"].plot(kind="bar")
        plt.xlabel("Disagreement example")
        plt.ylabel("Informative top score - baseline top score")
        plt.title("Informativeness gain in strongest model disagreements")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "disagreement_informativeness_gain.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-impressions",
        type=int,
        default=3000,
        help="Number of dev impressions to score. Use 0 for all.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top and bottom articles to save per model per impression.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_impressions = None if args.max_impressions == 0 else args.max_impressions

    model_df, disagreement_df, label_disagreement_df = make_demo_rows(
        max_impressions=max_impressions,
        top_n=args.top_n,
    )

    model_df.to_csv(OUT_DIR / "top_and_bottom_articles_by_model.csv", index=False)
    disagreement_df.to_csv(OUT_DIR / "model_disagreement_examples.csv", index=False)
    label_disagreement_df.to_csv(OUT_DIR / "label_disagreement_examples.csv", index=False)

    write_markdown_summary(model_df, disagreement_df, label_disagreement_df)
    make_plots(model_df, disagreement_df)

    print("\nSaved demo outputs:")
    print(f"- {OUT_DIR / 'top_and_bottom_articles_by_model.csv'}")
    print(f"- {OUT_DIR / 'model_disagreement_examples.csv'}")
    print(f"- {OUT_DIR / 'label_disagreement_examples.csv'}")
    print(f"- {OUT_DIR / 'demo_examples.md'}")
    print(f"- {OUT_DIR / 'top_recommendation_informativeness_boxplot.png'}")
    print(f"- {OUT_DIR / 'model_score_scatter.png'}")
    print(f"- {OUT_DIR / 'disagreement_informativeness_gain.png'}")


if __name__ == "__main__":
    main()
    