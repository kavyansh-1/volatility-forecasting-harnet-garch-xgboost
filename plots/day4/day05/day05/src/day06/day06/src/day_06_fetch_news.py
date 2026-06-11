import os
import time
import json
import requests 
import pandas as pd
from datetime import datetime , timedelta
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
BASE_URL = "https://newsapi.org/v2/everything"
TICKERS = ["SPY" , "QQQ" , "AAPL"]

TICKER_QUERY = {
    "SPY" : "S & P STOCK MARKET",
    "QQQ" : "Nasdaq technology Stocks",
    "AAPL" : "Apple Inc Stock earnings",

}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR , "output")
os.makedirs(OUT_DIR , exist_ok = True)

LOOKBACK_DAYS = 30

SYNTHETIC_HEADLINES = {
    "SPY": [
        "S&P 500 rallies as Fed signals rate pause",
        "Markets surge on strong jobs data",
        "Wall Street falls on recession fears",
        "S&P 500 hits record high amid tech rally",
        "Stocks drop sharply on inflation data",
        "Investors cautious ahead of Fed meeting",
        "Market volatility spikes on banking concerns",
        "S&P 500 recovers losses after earnings beat",
        "Broad market selloff on geopolitical tensions",
        "Stocks gain as consumer confidence rises",
    ],
    "QQQ": [
        "Nasdaq surges as big tech earnings beat estimates",
        "Tech stocks fall on rising interest rate fears",
        "AI boom drives Nasdaq to new highs",
        "Semiconductor stocks drag Nasdaq lower",
        "Nasdaq recovers after sharp overnight decline",
        "Tech valuations under pressure as yields rise",
        "Cloud stocks rally on strong quarterly results",
        "Nasdaq drops on disappointing guidance from megacaps",
        "Growth stocks rebound as inflation cools",
        "Tech sector leads market recovery",
    ],
    "AAPL": [
        "Apple beats earnings estimates on services growth",
        "Apple stock falls after weak iPhone sales outlook",
        "Apple announces record share buyback program",
        "iPhone demand concerns weigh on Apple shares",
        "Apple Vision Pro launch drives stock higher",
        "Apple cuts production forecast amid slowdown",
        "Analysts raise Apple price targets after results",
        "Apple faces regulatory pressure in Europe",
        "Apple stock rebounds after analyst upgrade",
        "Supply chain issues cloud Apple's outlook",
    ],
}

def fetch_from_newsapi(ticker:str , days_back: int = 30)-> list[dict]:
    if not NEWSAPI_KEY:
        return []
    
    query = TICKER_QUERY.get(ticker , ticker)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days = min(days_back , 29))

    params = { 
        "q" : query, 
        "from" : start_date.strftime("%Y-%m-%d"),
        "to" : end_date.strftime("%Y-%m-%d"),
        "language" : "en", 
        "sortby" : "publishedAt",
        "pageSize" : 100, 
        "apiKey" : NEWSAPI_KEY,
    }

    try: 
        resp = requests.get(BASE_URL , params=params , timeout = 10)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        time.sleep(1)
        print(f"{ticker}: fetched {len(articles)} articles from NewsAPI")
        return articles
    
    except requests.exceptions.RequestException as e:
        print(f"{ticker}: NewsAPI error - {e} ")
        return []
    
def build_synthetic_data(ticker:str, days_back: int = 30)-> list[dict]:
    import random
    random.seed(42)

    headlines = SYNTHETIC_HEADLINES.get(ticker , ["Market News Today"])
    end_date = datetime.utcnow()
    articles = []

    for i in range(days_back):
        date = end_date - timedelta(days = i)
        for _ in range(random.randint(1, 3)):
            headline = random.choice(headlines)
            articles.append({
                "title" : headline,
                "description" : headline + ". Full Story developing...",
                "publishedAt" : date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source" : {"name" : "Synthetic"}
            })
    print(f"{ticker}: generated {len(articles)} synthetic articles")
    return articles
        
def parse_articles(articles: list[dict], ticker: str)-> pd.DataFrame:
    rows = []
    for art in articles: 
        title = art.get("title", "") or ""
        desc = art.get("description" , "") or ""
        pub = art.get("publishedAt" , "")
        src = art.get("source", {}).get("name", "unknown")

        if not title or title == "[Removed]":
            continue

        text = (title + ". " + desc).strip()

        try:
            date = pd.to_datetime(pub).date()
        except Exception:
            continue

        rows.append({
            "date" : date, 
            "ticker" : ticker,
            "title" : title,
            "text" : text,
            "source" : src,
        })

    _df = pd.DataFrame(rows)
    if not _df.empty:
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.sort_values("date").reset_index(drop = True)
    return _df

def fetch_all_tickers(tickers: list = None, days_back: int = LOOKBACK_DAYS)-> pd.DataFrame:

    if tickers is None:
        tickers = TICKERS
    
    use_synthetic = not bool(NEWSAPI_KEY)
    if use_synthetic:
        print(" ⚠ NEWSAPI_KEY not set — using synthetic headlines")
        print(" Set NEWSAPI_KEY in a .env file for real data")

    all_dfs = []
    for ticker in tickers:
        if use_synthetic:
            articles = build_synthetic_data(ticker , days_back)
        else:
            articles = fetch_from_newsapi(ticker , days_back)
            if not articles:
                print(f" {ticker}: falling back to synthetic")
                articles = build_synthetic_data(ticker , days_back)
        
        df = parse_articles(articles, ticker)
        all_dfs.append(df)
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df
