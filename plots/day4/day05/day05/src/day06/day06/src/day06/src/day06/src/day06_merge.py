# ─────────────────────────────────────────────────────────────
# day06_merge.py
# Merges daily sentiment features with the existing
# volatility pipeline (processed CSVs from Day 2).
# Produces a sentiment-enriched DataFrame per ticker,
# ready for use in Days 7+ modelling.
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR   = os.path.join(BASE_DIR, "output")
DATA_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")

TICKERS   = ["SPY", "QQQ", "AAPL"]

SENT_COLS = [
    "sent_pos_mean",
    "sent_neg_mean",
    "sent_neu_mean",
    "sent_compound_mean",
    "sent_compound_std",
    "n_articles",
    "sent_pos_max",
    "sent_neg_max",
    "sent_roll3",
    "sent_roll7",
]


def load_volatility_data(ticker: str) -> pd.DataFrame:
    """
    Load the processed CSV from Day 2 for one ticker.
    Returns DataFrame with DatetimeIndex.
    """
    path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Run Day 2 pipeline first to generate processed CSVs."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def load_sentiment_data(ticker: str) -> pd.DataFrame:
    """
    Load daily sentiment CSV and filter to one ticker.
    Returns DataFrame with 'date' column as DatetimeIndex.
    """
    path = os.path.join(OUT_DIR, "day06_daily_sentiment.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Run day06_fetch_news.py + day06_sentiment.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["ticker"] == ticker].copy()
    df = df.set_index("date")
    df.index.name = "Date"
    return df


def merge_sentiment_with_vol(vol_df:  pd.DataFrame,
                              sent_df: pd.DataFrame,
                              ticker:  str) -> pd.DataFrame:
    """
    Left-join volatility data with sentiment on the date index.
    Strategy:
        - Left join: keep ALL volatility rows
        - Sentiment missing for a day → fill with forward-fill then zeros
        - Forward-fill assumes "no new news = same sentiment as yesterday"
        - Zero-fill for remaining NaN (start of series with no prior news)

    WHY LEFT JOIN?
    The volatility data is the primary series. News coverage is sparse
    (markets trade every day but news doesn't cover every ticker every day).
    We never want to lose a volatility row just because there was no news.
    """
    # Keep only sentiment columns we want
    sent_cols = [c for c in SENT_COLS if c in sent_df.columns]
    sent_sub  = sent_df[sent_cols]

    merged = vol_df.join(sent_sub, how="left")

    # Forward-fill: "no news today" → carry yesterday's sentiment
    merged[sent_cols] = merged[sent_cols].fillna(method="ffill", limit=3)

    # Zero-fill remaining: beginning of series before any news
    merged[sent_cols] = merged[sent_cols].fillna(0.0)

    # Flag which rows have real vs filled sentiment
    merged["has_real_sentiment"] = ~vol_df.index.isin(
        sent_sub.index[sent_sub.index.isin(vol_df.index)]
    )
    merged["has_real_sentiment"] = merged["has_real_sentiment"].astype(int)

    return merged


def compute_sentiment_stats(merged_df: pd.DataFrame,
                              ticker:   str) -> dict:
    """
    Quick summary stats on the merged data.
    Useful for spotting coverage gaps before modelling.
    """
    n_total   = len(merged_df)
    n_covered = (merged_df["n_articles"] > 0).sum()
    coverage  = n_covered / n_total * 100

    sent      = merged_df["sent_compound_mean"]
    return {
        "ticker"           : ticker,
        "total_days"       : n_total,
        "days_with_news"   : int(n_covered),
        "coverage_pct"     : round(coverage, 2),
        "mean_compound"    : round(sent.mean(), 4),
        "std_compound"     : round(sent.std(),  4),
        "min_compound"     : round(sent.min(),  4),
        "max_compound"     : round(sent.max(),  4),
        "pct_positive_days": round((sent > 0.1).mean() * 100, 2),
        "pct_negative_days": round((sent < -0.1).mean() * 100, 2),
    }


def run_merge_pipeline() -> dict:
    """
    Merge sentiment with volatility for all tickers.
    Saves one enriched CSV per ticker to output/.
    Returns dict of {ticker: merged_df}.
    """
    print(f"\n{'='*55}")
    print("  DAY 6 — Merging Sentiment + Volatility Data")
    print(f"{'='*55}")

    results    = {}
    stats_rows = []

    for ticker in TICKERS:
        print(f"\n  Processing {ticker}...")

        try:
            vol_df  = load_volatility_data(ticker)
            sent_df = load_sentiment_data(ticker)
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            continue

        merged = merge_sentiment_with_vol(vol_df, sent_df, ticker)

        # Save enriched CSV
        out_path = os.path.join(
            OUT_DIR, f"day06_{ticker}_sentiment_vol.csv"
        )
        merged.to_csv(out_path)
        print(f"  ✓ Saved → {out_path}")
        print(f"    Shape: {merged.shape}  "
              f"Columns: {merged.shape[1]}")

        stats = compute_sentiment_stats(merged, ticker)
        stats_rows.append(stats)
        print(f"    Coverage: {stats['coverage_pct']}%  "
              f"Mean compound: {stats['mean_compound']:.4f}")

        results[ticker] = merged

    # Save summary stats
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(OUT_DIR, "day06_sentiment_coverage.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  ✓ Coverage stats → {stats_path}")
    print(stats_df.to_string(index=False))

    return results


if __name__ == "__main__":
    run_merge_pipeline()