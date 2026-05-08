"""
Combines LLM informativeness scores with behavioral quality signals into Ỹ.

The raw BART-large-mnli score has poor absolute calibration: nearly all articles
score above 0.87 regardless of content. The relative ordering is meaningful,
so we convert raw scores to within-corpus percentile ranks before combining.

Formula: y_tilde = 0.5 * llm_score_pct_rank + 0.5 * behavioral_quality

Percentile rank is computed over all 51K articles so that a given article
always gets the same normalized score regardless of which articles were clicked.
This means:
  - Genuine click (behavioral=1) on 90th-pct article  → y_tilde = 0.95
  - Impulsive click (behavioral=0) on 90th-pct article → y_tilde = 0.45
  - Genuine click on 10th-pct article                  → y_tilde = 0.55
  - Impulsive click on 10th-pct article                → y_tilde = 0.05

Non-clicked articles are treated as y_tilde=0 by the training loop.

Output columns: UserID, NewsID, llm_score, llm_score_pct_rank, behavioral_quality, y_tilde
"""

import pandas as pd

from src.config import ARTICLE_SCORES_CSV, CLICK_QUALITY_CSV, Y_TILDE_CSV

LLM_WEIGHT = 0.5
BEHAVIORAL_WEIGHT = 0.5


def main():
    print("Loading scores...")
    llm = pd.read_csv(ARTICLE_SCORES_CSV)
    behavioral = pd.read_csv(CLICK_QUALITY_CSV)

    print(f"  LLM scores:        {len(llm):,} articles")
    print(f"  Behavioral clicks: {len(behavioral):,} click events")

    # Rank-normalize over the full article corpus so relative ordering is
    # preserved but the [0,1] range is fully utilized.
    llm["llm_score_pct_rank"] = llm["informativeness_score"].rank(pct=True).round(4)

    print(f"  LLM raw  — mean: {llm['informativeness_score'].mean():.3f}, "
          f"std: {llm['informativeness_score'].std():.3f}")
    print(f"  LLM rank — mean: {llm['llm_score_pct_rank'].mean():.3f}, "
          f"std: {llm['llm_score_pct_rank'].std():.3f}")

    # Join on NewsID; only clicked articles are in behavioral
    merged = behavioral.merge(
        llm[["NewsID", "informativeness_score", "llm_score_pct_rank"]],
        on="NewsID",
        how="left",
    )

    # Articles not found in news.tsv get corpus median rank (0.5)
    missing = merged["llm_score_pct_rank"].isna().sum()
    if missing:
        print(f"  {missing} clicks missing LLM score — filling rank with 0.5 (median)")
    merged["llm_score_pct_rank"] = merged["llm_score_pct_rank"].fillna(0.5)
    merged["informativeness_score"] = merged["informativeness_score"].fillna(
        llm["informativeness_score"].mean()
    )

    merged["y_tilde"] = (
        LLM_WEIGHT * merged["llm_score_pct_rank"] +
        BEHAVIORAL_WEIGHT * merged["quality"]
    ).round(4)

    output = merged[["UserID", "NewsID", "informativeness_score", "llm_score_pct_rank",
                      "quality", "y_tilde"]]
    output = output.rename(columns={
        "informativeness_score": "llm_score",
        "quality": "behavioral_quality",
    })
    output.to_csv(Y_TILDE_CSV, index=False)

    print(f"\nDone. {len(output):,} rows saved to {Y_TILDE_CSV}")
    print(f"  y_tilde mean:  {output['y_tilde'].mean():.3f}")
    print(f"  y_tilde std:   {output['y_tilde'].std():.3f}")
    print(f"  y_tilde range: [{output['y_tilde'].min():.3f}, {output['y_tilde'].max():.3f}]")


if __name__ == "__main__":
    main()
