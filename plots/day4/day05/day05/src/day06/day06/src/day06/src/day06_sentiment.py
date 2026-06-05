# ─────────────────────────────────────────────────────────────
# day06_sentiment.py
# Scores each news article using FinBERT — a BERT model
# fine-tuned on financial text. Produces three scores per
# article: positive, negative, neutral (sum to 1.0).
# Aggregates to daily ticker-level sentiment features.
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")

# FinBERT — fine-tuned on financial phrases/news
# Downloads ~440MB on first run, then cached locally
FINBERT_MODEL = "ProsusAI/finbert"

# Max tokens FinBERT accepts (BERT limit is 512)
MAX_TOKENS = 512

# Batch size for inference — reduce if GPU OOM
BATCH_SIZE = 16


# ── Model loading ───────────────────────────────────────────────
def load_finbert(device: torch.device = None):
    """
    Load FinBERT tokenizer and model.
    Downloads from HuggingFace Hub on first call, cached after.
    Returns (tokenizer, model, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Loading FinBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model     = model.to(device)
    model.eval()   # inference only — no gradient tracking needed
    print(f"  FinBERT loaded. Labels: {model.config.id2label}")
    return tokenizer, model, device


# ── Batch inference ─────────────────────────────────────────────
@torch.no_grad()
def score_batch(texts:     list[str],
                tokenizer,
                model,
                device:    torch.device) -> np.ndarray:
    """
    Score a batch of text strings with FinBERT.
    Returns numpy array of shape (len(texts), 3):
        column 0 = positive probability
        column 1 = negative probability
        column 2 = neutral  probability
    (FinBERT label order: positive=0, negative=1, neutral=2)
    """
    # Tokenize: truncate to MAX_TOKENS, pad shorter texts to same length
    encoding = tokenizer(
        texts,
        padding      = True,
        truncation   = True,
        max_length   = MAX_TOKENS,
        return_tensors = "pt",   # return PyTorch tensors
    )
    # Move input tensors to same device as model
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    logits = model(input_ids      = input_ids,
                   attention_mask = attention_mask).logits

    # Softmax converts raw logits → probabilities that sum to 1
    probs = softmax(logits, dim=-1).cpu().numpy()
    return probs   # shape: (batch_size, 3)


def score_articles(df:        pd.DataFrame,
                   tokenizer,
                   model,
                   device:    torch.device,
                   text_col:  str = "text") -> pd.DataFrame:
    """
    Score every article in the DataFrame using FinBERT.
    Processes in batches to avoid memory overflow.
    Adds three new columns: sent_pos, sent_neg, sent_neu.
    Also adds compound score: sent_pos - sent_neg  ∈ (-1, +1)
    """
    texts      = df[text_col].fillna("").tolist()
    n          = len(texts)
    all_probs  = []

    for start in tqdm(range(0, n, BATCH_SIZE),
                      desc="  Scoring articles",
                      unit="batch"):
        batch = texts[start : start + BATCH_SIZE]
        probs = score_batch(batch, tokenizer, model, device)
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)   # (n_articles, 3)

    df = df.copy()
    # FinBERT label order: 0=positive, 1=negative, 2=neutral
    df["sent_pos"] = all_probs[:, 0]
    df["sent_neg"] = all_probs[:, 1]
    df["sent_neu"] = all_probs[:, 2]

    # Compound: net sentiment direction
    # +1 = perfectly positive, -1 = perfectly negative
    df["sent_compound"] = df["sent_pos"] - df["sent_neg"]

    return df


# ── Daily aggregation ───────────────────────────────────────────
def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse article-level scores to daily ticker-level features.

    For each (date, ticker) pair computes:
        sent_pos_mean    : mean positive score across all articles
        sent_neg_mean    : mean negative score
        sent_neu_mean    : mean neutral score
        sent_compound_mean : mean compound score
        sent_compound_std  : std of compound (sentiment uncertainty/disagreement)
        n_articles         : how many articles that day
        sent_pos_max       : most positive article score (extreme event detector)
        sent_neg_max       : most negative article score

    WHY AGGREGATE TO DAILY?
    Models are trained at daily frequency. Having one sentiment row
    per (date, ticker) makes merging with price/volatility data trivial.
    """
    agg = df.groupby(["date", "ticker"]).agg(
        sent_pos_mean     = ("sent_pos",      "mean"),
        sent_neg_mean     = ("sent_neg",      "mean"),
        sent_neu_mean     = ("sent_neu",      "mean"),
        sent_compound_mean= ("sent_compound", "mean"),
        sent_compound_std = ("sent_compound", "std"),
        n_articles        = ("sent_compound", "count"),
        sent_pos_max      = ("sent_pos",      "max"),
        sent_neg_max      = ("sent_neg",      "max"),
    ).reset_index()

    # Fill std=NaN for days with only 1 article
    agg["sent_compound_std"] = agg["sent_compound_std"].fillna(0.0)

    # Rolling 3-day and 7-day mean compound score per ticker
    # Gives the model a smoothed sentiment signal without day-to-day noise
    agg = agg.sort_values(["ticker", "date"]).reset_index(drop=True)

    smoothed = []
    for ticker in agg["ticker"].unique():
        mask = agg["ticker"] == ticker
        sub  = agg[mask].copy()
        sub["sent_roll3"] = (sub["sent_compound_mean"]
                              .rolling(3, min_periods=1).mean())
        sub["sent_roll7"] = (sub["sent_compound_mean"]
                              .rolling(7, min_periods=1).mean())
        smoothed.append(sub)

    agg = pd.concat(smoothed, ignore_index=True)
    return agg


# ── Main ────────────────────────────────────────────────────────
def run_sentiment_pipeline(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: load model → score → aggregate → save.
    Returns daily sentiment DataFrame.
    """
    tokenizer, model, device = load_finbert()

    print(f"\n  Scoring {len(articles_df)} articles...")
    scored_df = score_articles(articles_df, tokenizer, model, device)

    # Save article-level scores
    art_path = os.path.join(OUT_DIR, "day06_article_scores.csv")
    scored_df.drop(columns=["text"], errors="ignore").to_csv(
        art_path, index=False
    )
    print(f"  ✓ Article scores → {art_path}")

    # Aggregate to daily
    daily_df = aggregate_daily_sentiment(scored_df)

    daily_path = os.path.join(OUT_DIR, "day06_daily_sentiment.csv")
    daily_df.to_csv(daily_path, index=False)
    print(f"  ✓ Daily sentiment → {daily_path}")

    return daily_df