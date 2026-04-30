"""
Combines LLM informativeness scores (article_scores.csv) with behavioral
quality signals (click_quality.csv) into a single Ỹ label (y_tilde.csv).

Output columns: UserID, NewsID, llm_score, behavioral_quality, y_tilde
y_tilde = 0.5 * llm_score + 0.5 * behavioral_quality (equal weighting)

Only covers clicked articles (rows in click_quality.csv). The training loop
treats non-clicked articles as y_tilde=0.
"""

import pandas as pd

from src.config import ARTICLE_SCORES_CSV, CLICK_QUALITY_CSV, Y_TILDE_CSV

LLM_WEIGHT = 0.5
BEHAVIORAL_WEIGHT = 0.5


def main():
    print("Loading scores...")
    llm = pd.read_csv(ARTICLE_SCORES_CSV)
    behavioral = pd.read_csv(CLICK_QUALITY_CSV)

    print(f"  LLM scores:       {len(llm):,} articles")
    print(f"  Behavioral clicks: {len(behavioral):,} click events")

    # Join on NewsID
    merged = behavioral.merge(llm, on="NewsID", how="left")

    # Fill missing LLM scores (articles not in news.tsv) with dataset mean
    mean_llm = llm["informativeness_score"].mean()
    missing = merged["informativeness_score"].isna().sum()
    if missing:
        print(f"  {missing} clicks missing LLM score — filling with mean ({mean_llm:.3f})")
    merged["informativeness_score"] = merged["informativeness_score"].fillna(mean_llm)

    # Combine
    merged["y_tilde"] = (
        LLM_WEIGHT * merged["informativeness_score"] +
        BEHAVIORAL_WEIGHT * merged["quality"]
    ).round(4)

    output = merged[["UserID", "NewsID", "informativeness_score", "quality", "y_tilde"]]
    output = output.rename(columns={"informativeness_score": "llm_score",
                                    "quality": "behavioral_quality"})
    output.to_csv(Y_TILDE_CSV, index=False)

    print(f"\nDone. {len(output):,} rows saved to {Y_TILDE_CSV}")
    print(f"  y_tilde mean:  {output['y_tilde'].mean():.3f}")
    print(f"  y_tilde std:   {output['y_tilde'].std():.3f}")
    print(f"  y_tilde range: [{output['y_tilde'].min():.3f}, {output['y_tilde'].max():.3f}]")


if __name__ == "__main__":
    main()
