"""Streamlit demo for side-by-side comparison of click-trained vs informative recommender."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import (
    ARTICLE_SCORES_CSV,
    BASELINE_MODEL,
    CLICK_QUALITY_CSV,
    DEV_DIR,
    INFORMATIVE_MODEL,
    TRAIN_DIR,
    Y_TILDE_CSV,
)
from src.models.embeddings import (
    BEHAVIOR_COLS,
    EMBEDDING_DIM,
    NEWS_COLS,
    RecommenderMLP,
    load_embeddings,
)


@st.cache_resource
def load_model(path: str) -> RecommenderMLP:
    model = RecommenderMLP()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data
def load_news() -> pd.DataFrame:
    frames = []
    for p in [TRAIN_DIR / "news.tsv", DEV_DIR / "news.tsv"]:
        if p.exists():
            df = pd.read_csv(p, sep="\t", names=NEWS_COLS, dtype=str).fillna("")
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=NEWS_COLS)

    news = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["NewsID"])
    return news


@st.cache_data
def load_behaviors() -> pd.DataFrame:
    path = DEV_DIR / "behaviors.tsv"
    if not path.exists():
        return pd.DataFrame(columns=BEHAVIOR_COLS)
    return pd.read_csv(path, sep="\t", names=BEHAVIOR_COLS, dtype=str).dropna(subset=["Impressions"])


@st.cache_data
def load_article_scores() -> Dict[str, float]:
    if not ARTICLE_SCORES_CSV.exists():
        return {}
    df = pd.read_csv(ARTICLE_SCORES_CSV)
    if "NewsID" not in df.columns:
        return {}

    score_col = "informativeness_score" if "informativeness_score" in df.columns else None
    if score_col is None:
        numeric_cols = [
            c
            for c in df.columns
            if c != "NewsID" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return {}
        score_col = numeric_cols[0]

    return dict(zip(df["NewsID"].astype(str), df[score_col].astype(float)))


@st.cache_data
def load_user_y_tilde() -> Dict[Tuple[str, str], float]:
    if not Y_TILDE_CSV.exists():
        return {}
    df = pd.read_csv(Y_TILDE_CSV)
    needed = {"UserID", "NewsID", "y_tilde"}
    if not needed.issubset(df.columns):
        return {}
    return {
        (str(r.UserID), str(r.NewsID)): float(r.y_tilde)
        for r in df.itertuples(index=False)
    }


@st.cache_data
def load_click_quality() -> Dict[Tuple[str, str], float]:
    if not CLICK_QUALITY_CSV.exists():
        return {}
    df = pd.read_csv(CLICK_QUALITY_CSV)
    needed = {"UserID", "NewsID", "quality"}
    if not needed.issubset(df.columns):
        return {}
    return {
        (str(r.UserID), str(r.NewsID)): float(r.quality)
        for r in df.itertuples(index=False)
    }


def parse_impressions(impressions: str) -> Tuple[List[str], List[int]]:
    news_ids, clicks = [], []
    for item in str(impressions).split():
        try:
            nid, c = item.rsplit("-", 1)
            news_ids.append(nid)
            clicks.append(int(c))
        except ValueError:
            continue
    return news_ids, clicks


def build_user_embedding(history: str, emb: Dict[str, np.ndarray]) -> np.ndarray:
    if not isinstance(history, str) or not history.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    vecs = [emb[nid] for nid in history.split() if nid in emb]
    if not vecs:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return np.mean(vecs, axis=0).astype(np.float32)


def score_candidates(
    model: RecommenderMLP,
    user_emb: np.ndarray,
    candidate_ids: List[str],
    emb: Dict[str, np.ndarray],
) -> np.ndarray:
    article_mat = np.stack([emb[nid] for nid in candidate_ids]).astype(np.float32)
    user_mat = np.repeat(user_emb[None, :], len(candidate_ids), axis=0).astype(np.float32)

    with torch.inference_mode():
        scores = model(torch.from_numpy(user_mat), torch.from_numpy(article_mat)).numpy()

    return scores.astype(float)


def build_recs_table(
    ranked_ids: List[str],
    scores: np.ndarray,
    click_map: Dict[str, int],
    user_id: str,
    news_map: Dict[str, dict],
    info_scores: Dict[str, float],
    y_tilde: Dict[Tuple[str, str], float],
    quality: Dict[Tuple[str, str], float],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for nid, score in zip(ranked_ids[:top_k], scores[:top_k]):
        meta = news_map.get(nid, {})
        rows.append(
            {
                "NewsID": nid,
                "Title": meta.get("Title", ""),
                "Category": meta.get("Category", ""),
                "ModelScore": float(score),
                "Clicked": int(click_map.get(nid, 0)),
                "LLMInfoScore": info_scores.get(nid, np.nan),
                "y_tilde": y_tilde.get((user_id, nid), np.nan),
                "ClickQuality": quality.get((user_id, nid), np.nan),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Informative News Recommender", layout="wide")

    st.title("Informative News Recommender Demo")
    st.caption("6.C395 final project: compare click-optimized vs informativeness-optimized recommendations.")

    with st.expander("Project framing", expanded=False):
        st.markdown(
            """
- Baseline objective: optimize click label `Y`.
- Proposed objective: optimize informativeness proxy `Ỹ`.
- Label-bias view: `Y = Ỹ + Δ`, where `Δ` captures impulsive or outrage-driven clicks.
- This demo scores the same user impression with both trained models and compares top-K lists.
            """
        )

    if not BASELINE_MODEL.exists() or not INFORMATIVE_MODEL.exists():
        st.error("Missing trained models. Run Stage 2 training first.")
        return

    if not (DEV_DIR / "behaviors.tsv").exists() or not (DEV_DIR / "news.tsv").exists():
        st.error("Missing MIND dev split. Download MIND-small and run scripts/download_data.py.")
        return

    emb = load_embeddings()
    baseline_model = load_model(str(BASELINE_MODEL))
    informative_model = load_model(str(INFORMATIVE_MODEL))

    behaviors = load_behaviors()
    news = load_news()
    news_map = news.set_index("NewsID").to_dict("index") if not news.empty else {}
    info_scores = load_article_scores()
    y_tilde = load_user_y_tilde()
    quality = load_click_quality()

    st.sidebar.header("Demo controls")
    top_k = st.sidebar.slider("Top-K recommendations", min_value=5, max_value=20, value=10)
    sample_size = st.sidebar.slider("Users shown", min_value=100, max_value=5000, value=1000, step=100)

    sampled = behaviors.head(sample_size).copy()
    users = sorted(sampled["UserID"].dropna().unique().tolist())
    if not users:
        st.error("No valid user impressions found in dev behaviors.")
        return

    user_id = st.sidebar.selectbox("User", users)
    user_rows = sampled[sampled["UserID"] == user_id].reset_index(drop=True)
    imp_idx = st.sidebar.selectbox("Impression index for user", list(range(len(user_rows))))

    row = user_rows.iloc[imp_idx]
    candidate_ids_raw, candidate_clicks = parse_impressions(row["Impressions"])

    valid_pairs = [(nid, c) for nid, c in zip(candidate_ids_raw, candidate_clicks) if nid in emb]
    if not valid_pairs:
        st.warning("No candidate articles from this impression have embeddings.")
        return

    candidate_ids = [nid for nid, _ in valid_pairs]
    click_map = {nid: c for nid, c in valid_pairs}

    user_emb = build_user_embedding(row["History"] if pd.notna(row["History"]) else "", emb)
    base_scores = score_candidates(baseline_model, user_emb, candidate_ids, emb)
    info_model_scores = score_candidates(informative_model, user_emb, candidate_ids, emb)

    base_order = np.argsort(-base_scores)
    info_order = np.argsort(-info_model_scores)

    base_ranked_ids = [candidate_ids[i] for i in base_order]
    info_ranked_ids = [candidate_ids[i] for i in info_order]

    base_table = build_recs_table(
        base_ranked_ids,
        base_scores[base_order],
        click_map,
        user_id,
        news_map,
        info_scores,
        y_tilde,
        quality,
        top_k,
    )
    info_table = build_recs_table(
        info_ranked_ids,
        info_model_scores[info_order],
        click_map,
        user_id,
        news_map,
        info_scores,
        y_tilde,
        quality,
        top_k,
    )

    overlap = len(set(base_table["NewsID"]).intersection(set(info_table["NewsID"])))
    base_llm_mean = float(base_table["LLMInfoScore"].mean()) if not base_table.empty else float("nan")
    info_llm_mean = float(info_table["LLMInfoScore"].mean()) if not info_table.empty else float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates in impression", len(candidate_ids))
    m2.metric(f"Top-{top_k} overlap", overlap)
    m3.metric("Baseline top-K avg LLM info", f"{base_llm_mean:.3f}" if np.isfinite(base_llm_mean) else "n/a")
    m4.metric("Informative top-K avg LLM info", f"{info_llm_mean:.3f}" if np.isfinite(info_llm_mean) else "n/a")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Baseline model (trained on clicks Y)")
        st.dataframe(base_table, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Informative model (trained on Ỹ)")
        st.dataframe(info_table, use_container_width=True, hide_index=True)

    st.subheader("Interpretation")
    delta = info_llm_mean - base_llm_mean if np.isfinite(info_llm_mean) and np.isfinite(base_llm_mean) else np.nan
    if np.isfinite(delta):
        st.write(
            f"For this impression, the informative model's top-{top_k} list has "
            f"{delta:+.3f} higher average LLM informativeness score than the baseline."
        )
    else:
        st.write("Not enough LLM scores were available to compute a reliable per-impression delta.")


if __name__ == "__main__":
    main()
