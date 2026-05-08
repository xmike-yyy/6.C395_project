from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import (
    TRAIN_DIR, DEV_DIR, ARTICLE_SCORES_CSV, CLICK_QUALITY_CSV, Y_TILDE_CSV,
    BASELINE_MODEL, INFORMATIVE_MODEL,
)
from src.models.embeddings import (
    BEHAVIOR_COLS, NEWS_COLS, EMBEDDING_DIM, RecommenderMLP, load_embeddings,
)

def load_model(path: Path) -> RecommenderMLP:
    state = torch.load(path, map_location="cpu")
    model = RecommenderMLP()
    model.load_state_dict(state)
    model.eval()
    return model

def load_behaviors() -> pd.DataFrame:
    behaviors_path = DEV_DIR / "behaviors.tsv"
    df = pd.read_csv(behaviors_path, sep="\t", names=BEHAVIOR_COLS, dtype=str).dropna(subset=["Impressions"])
    return df

def load_metadata() -> Dict[str, Dict[str,str]]:
    meta: Dict[str, Dict[str,str]] = {}
    for path in [TRAIN_DIR/"news.tsv", DEV_DIR/"news.tsv"]:
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", names=NEWS_COLS, dtype=str).fillna("")
        for row in df.itertuples(index=False):
            meta[row.NewsID] = {
                "title": str(row.Title), "abstract": str(row.Abstract),
                "category": str(row.Category), "subcategory": str(row.SubCategory),
                "url": str(row.URL)
            }
    return meta

def load_article_scores() -> Dict[str, float]:
    if not ARTICLE_SCORES_CSV.exists():
        return {}
    df = pd.read_csv(ARTICLE_SCORES_CSV)
    # find the column for informativeness score
    for col in ["article_score","informativeness_score","score"]:
        if col in df.columns:
            score_col = col; break
    else:
        num_cols = [c for c in df.columns if c not in ("NewsID",) and pd.api.types.is_numeric_dtype(df[c])]
        score_col = num_cols[0] if num_cols else None
    if score_col is None: 
        return {}
    return dict(zip(df.NewsID.astype(str), df[score_col].astype(float)))

def load_y_tilde() -> Dict[Tuple[str, str], float]:
    if not Y_TILDE_CSV.exists():
        return {}
    df = pd.read_csv(Y_TILDE_CSV)
    if not {"UserID", "NewsID", "y_tilde"}.issubset(df.columns):
        return {}
    return {(str(r.UserID), str(r.NewsID)): float(r.y_tilde)
            for r in df.itertuples(index=False)}

def load_click_quality() -> Tuple[Dict[Tuple[str,str],float], Dict[str,float]]:
    if not CLICK_QUALITY_CSV.exists():
        return {}, {}
    df = pd.read_csv(CLICK_QUALITY_CSV)
    user_art = {}
    if {"UserID","NewsID"}.issubset(df.columns):
        user_art = { (str(r.UserID), str(r.NewsID)): float(r.quality)
                     for r in df.itertuples(index=False) }
    art = df.groupby("NewsID")["quality"].mean().astype(float).to_dict()
    return user_art, {str(k): float(v) for k,v in art.items()}

def parse_impressions(imps: str) -> Tuple[list[str], np.ndarray]:
    news_ids = []
    clicks = []
    for item in imps.split():
        try:
            nid, c = item.rsplit("-",1)
            news_ids.append(nid); clicks.append(int(c))
        except ValueError:
            continue
    return news_ids, np.array(clicks, dtype=np.float32)

def build_user_embedding(history: str, embeddings: Dict[str,np.ndarray]) -> np.ndarray:
    if not isinstance(history, str) or history.strip() == "":
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vecs = [embeddings[nid] for nid in history.split() if nid in embeddings]
    if not vecs:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def score_candidates(model: RecommenderMLP, user_emb: np.ndarray, 
                     news_ids: list[str], embeddings: Dict[str,np.ndarray]) -> np.ndarray:
    art_embs = np.stack([embeddings[n] for n in news_ids]).astype(np.float32)
    user_batch = np.repeat(user_emb[None,:], len(news_ids), axis=0).astype(np.float32)
    with torch.inference_mode():
        scores = model(torch.from_numpy(user_batch), torch.from_numpy(art_embs)).numpy()
    return scores.flatten()

def main():
    parser = argparse.ArgumentParser(description="Demo evaluation of model disagreements")
    parser.add_argument("--max-impressions", type=int, default=3000,
                        help="Max number of dev impressions to process (0=all)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top-N examples to select for each metric")
    args = parser.parse_args()

    baseline = load_model(BASELINE_MODEL)
    informative = load_model(INFORMATIVE_MODEL)
    embeddings = load_embeddings()
    behaviors = load_behaviors()
    metadata = load_metadata()
    art_scores = load_article_scores()
    y_tilde = load_y_tilde()
    user_art_quality, art_quality = load_click_quality()

    n_imps = len(behaviors) if args.max_impressions==0 else min(len(behaviors), args.max_impressions)
    behaviors = behaviors.iloc[:n_imps]

    records = []
    # Loop over impressions
    for row in behaviors.itertuples(index=False):
        user = str(row.UserID)
        history = row.History if pd.notna(row.History) else ""
        news_ids, click_labels = parse_impressions(row.Impressions)
        # Filter only IDs present in embeddings
        valid = [(nid, lab) for nid, lab in zip(news_ids, click_labels) if nid in embeddings]
        if len(valid) < 1: 
            continue
        news_ids, click_labels = zip(*valid)
        user_emb = build_user_embedding(history, embeddings)

        base_scores = score_candidates(baseline, user_emb, list(news_ids), embeddings)
        info_scores = score_candidates(informative, user_emb, list(news_ids), embeddings)

        # Compute metrics for each candidate
        for nid, click, bscore, iscore in zip(news_ids, click_labels, base_scores, info_scores):
            article_info = art_scores.get(nid, np.nan)
            yt = y_tilde.get((user, nid), np.nan)
            rec = {
                "UserID": user, "NewsID": nid,
                "Click": int(click),
                "ArticleInfo": float(article_info),
                "Y_tilde": float(yt) if not np.isnan(yt) else np.nan,
                "BaselineScore": float(bscore),
                "InformativeScore": float(iscore),
            }
            # gaps
            if not np.isnan(article_info):
                rec["Gap_Click_Info"] = abs(click - article_info)
            else:
                rec["Gap_Click_Info"] = np.nan
            if not np.isnan(yt):
                rec["Gap_Click_Yt"] = abs(click - yt)
            else:
                rec["Gap_Click_Yt"] = np.nan
            rec["ScoreGap"] = abs(bscore - iscore)
            rec["PerpDist"] = rec["ScoreGap"] / math.sqrt(2)
            records.append(rec)

    df = pd.DataFrame(records)
    # Select top examples
    top_info_gap = df.nlargest(args.top_n, "Gap_Click_Info").dropna()
    top_ytilde_gap = df.nlargest(args.top_n, "Gap_Click_Yt").dropna()
    top_score_gap = df.nlargest(args.top_n, "ScoreGap")

    # Save CSVs
    top_info_gap.to_csv("outputs/demo_disagreements/top_by_click_info_gap.csv", index=False)
    top_ytilde_gap.to_csv("outputs/demo_disagreements/top_by_click_ytilde_gap.csv", index=False)
    top_score_gap.to_csv("outputs/demo_disagreements/top_by_score_gap.csv", index=False)

    # Markdown summary
    with open("outputs/demo_disagreements/demo_disagreements.md", "w", encoding="utf-8") as md:
        md.write("# Top Disagreement Examples\n\n")
        md.write("**Highest click-vs-informativeness gaps:**\n")
        for r in top_info_gap.itertuples(index=False):
            title = metadata.get(r.NewsID, {}).get("title","")
            md.write(f"- (User {r.UserID}, Article {r.NewsID}) {title} "
                     f"(click={int(r.Click)}, info={r.ArticleInfo:.2f}, gap={r.Gap_Click_Info:.2f})\n")
        md.write("\n**Highest click-vs-yt̃ gaps:**\n")
        for r in top_ytilde_gap.itertuples(index=False):
            title = metadata.get(r.NewsID, {}).get("title","")
            md.write(f"- (User {r.UserID}, Article {r.NewsID}) {title} "
                     f"(click={int(r.Click)}, y_tilde={r.Y_tilde:.2f}, gap={r.Gap_Click_Yt:.2f})\n")
        md.write("\n**Largest model score gaps:**\n")
        for r in top_score_gap.itertuples(index=False):
            title = metadata.get(r.NewsID, {}).get("title","")
            md.write(f"- (User {r.UserID}, Article {r.NewsID}) {title} "
                     f"(scores: baseline={r.BaselineScore:.2f}, info={r.InformativeScore:.2f}, "
                     f"diff={r.ScoreGap:.2f})\n")

    # Plots
    # Scatter colored by click-info gap
    plt.figure(figsize=(6,5))
    cmap = plt.get_cmap("coolwarm")
    sc = plt.scatter(df["BaselineScore"], df["InformativeScore"], 
                     c=df["Gap_Click_Info"], cmap=cmap, alpha=0.6, s=20)
    plt.plot([0,1],[0,1],'k--', alpha=0.7)
    plt.colorbar(sc, label="|Click - ArticleInfo| gap")
    plt.xlabel("Baseline model score")
    plt.ylabel("Informative model score")
    plt.title("Score disagreement (red=high label gap)")
    # annotate top 5 score differences
    for idx in df.nlargest(5, "ScoreGap").index:
        x = df.loc[idx, "BaselineScore"]; y = df.loc[idx, "InformativeScore"]
        plt.annotate(df.loc[idx, "NewsID"], (x, y), textcoords="offset points", xytext=(2,2))
    plt.savefig("outputs/demo_disagreements/score_scatter.png", dpi=200)
    plt.close()

    # Boxplot of article informativeness in top-K lists
    K = 20
    top_base_idx = df.nlargest(K, "BaselineScore").index
    top_inf_idx  = df.nlargest(K, "InformativeScore").index
    base_infos = df.loc[top_base_idx, "ArticleInfo"].dropna()
    inf_infos  = df.loc[top_inf_idx, "ArticleInfo"].dropna()
    plt.figure(figsize=(5,4))
    plt.boxplot([base_infos, inf_infos], labels=["Baseline","Informative"])
    plt.title(f"Article informativeness of top-{K} items")
    plt.ylabel("Informativeness score")
    plt.savefig("outputs/demo_disagreements/informativeness_boxplot.png", dpi=200)
    plt.close()

    print("Disagreement analysis complete. Output saved to outputs/demo_disagreements/")

if __name__ == "__main__":
    main()
