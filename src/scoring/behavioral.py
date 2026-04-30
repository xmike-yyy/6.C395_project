"""
Extracts a depth-of-engagement signal from behaviors.tsv.

For each clicked article, checks whether the user's next click (same day)
is in the same category → quality=1 (genuine interest) or a different
category → quality=0 (shallow/impulsive click).

Outputs data/click_quality.csv with columns: UserID, NewsID, quality.
"""

import pandas as pd

from src.config import TRAIN_DIR, CLICK_QUALITY_CSV

BEHAVIOR_TSV = TRAIN_DIR / "behaviors.tsv"
NEWS_TSV = TRAIN_DIR / "news.tsv"

BEHAVIOR_COLS = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
NEWS_COLS = ["NewsID", "Category", "SubCategory", "Title", "Abstract",
             "URL", "TitleEntities", "AbstractEntities"]


def parse_clicks(behaviors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in behaviors.iterrows():
        if pd.isna(row["Impressions"]):
            continue
        for item in row["Impressions"].split():
            news_id, label = item.rsplit("-", 1)
            if label == "1":
                rows.append({
                    "UserID": row["UserID"],
                    "NewsID": news_id,
                    "Time": row["Time"],
                })
    return pd.DataFrame(rows)


def main():
    print("Loading news categories...")
    news_df = pd.read_csv(NEWS_TSV, sep="\t", names=NEWS_COLS,
                          usecols=["NewsID", "Category"])
    category_map = news_df.set_index("NewsID")["Category"].to_dict()

    print("Loading behaviors...")
    behaviors = pd.read_csv(BEHAVIOR_TSV, sep="\t", names=BEHAVIOR_COLS)
    behaviors["Time"] = pd.to_datetime(behaviors["Time"])

    print("Extracting clicks...")
    clicks = parse_clicks(behaviors)
    clicks["Date"] = clicks["Time"].dt.date
    clicks["Category"] = clicks["NewsID"].map(category_map)
    clicks = clicks.sort_values(["UserID", "Time"]).reset_index(drop=True)

    print("Computing quality signal...")
    # Vectorized: compare each click's category to the next click's category
    # within the same user and same day
    clicks["next_Category"] = clicks.groupby("UserID")["Category"].shift(-1)
    clicks["next_Date"] = clicks.groupby("UserID")["Date"].shift(-1)

    clicks["quality"] = (
        (clicks["Category"] == clicks["next_Category"]) &
        (clicks["Date"] == clicks["next_Date"])
    ).astype(int)

    output = clicks[["UserID", "NewsID", "quality"]]
    output.to_csv(CLICK_QUALITY_CSV, index=False)

    total = len(output)
    positive = clicks["quality"].sum()
    print(f"Done. {total} click events | {positive} genuine ({positive/total:.1%}) | "
          f"{total - positive} impulsive ({(total-positive)/total:.1%})")
    print(f"Saved to {CLICK_QUALITY_CSV}")


if __name__ == "__main__":
    main()
