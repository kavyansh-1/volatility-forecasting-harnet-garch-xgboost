import os , sys , time , warnings 
warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0 , os.path.dirname(__file__))
sys.path.insert(0 , os.path.join(os.path.dirname(__file__), ".." , ".." , "day16" , "src"))

from day17_leverage_effect import run_leverage_analysis
from day17_asymmetric_garch import run_asymmetric_garch
from day17_realized_moments import run_realized_moments
from day17_asymmetric_har import run_asymmetric_har
from day17_plots import run_all_plots

def load_intraday_data():
    try: 
        from day16_intraday_data import run_intraday_data_pipeline
        return run_intraday_data_pipeline()
    except ImportError:
        print("Day 16 module not found - generating fresh intraday data")
        from day17_realized_moments import TICKERS
        import pandas as pd 
        import numpy as np 

        DATA_DIR = os.path.join(BASE_DIR , ".." , "data" , "processed")
        BARS = 78

        def u_shape(n=78):
            t = np.linspace(-1,1,n)
            w = 1.0 + 0.5 * t ** 2
            return w / w.sum()

        results = {}
        for ticker in TICKERS:
            path = os.path.join(DATA_DIR , f"{ticker}_processed.csv")
            if not os.path.exists(path):
                continue 
            df = pd.read_csv(path , index_col = "dates" , parse_dates = True)
            rng = np.random.default_rng(42)
            w = u_shape(BARS)
            rows = []
            r = df["log_return"].dropna().values
            c = df["Close"].values
            for d_idx, date in enumerate(df.index[df["log_return"].notna()]):
                if d_idx>= len(r):
                    break
                bar_vols = np.sqrt(r[d_idx]**2 * w)
                rets = rng.normal(0, bar_vols , BARS)
                if abs(rets.sum()) > 1e-10:
                    rets = rets * (r[d_idx] / rets.sum())
                prices = (c[d_idx-1] if d_idx > 0 else c[0]) * np.exp(np.cumsum(rets))
                for i in range(BARS): 
                    rows.append({
                        "date" : str(date.date()), 
                        "bar" : i, 
                        "log_ret" : rets[i], 
                        "Close" : prices[i],
                    }) 
            intra = pd.DataFrame(rows)
            intra.index = pd.RangeIndex(len(intra))
            results[ticker] = intra 
        return results 
    
def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 17 — Leverage Effect, Asymmetric GARCH & HAR")
    print(f"{'='*55}")

    # Load intraday data (from Day 16 or fresh synthetic)
    print("\n[0/5] Loading intraday data...")
    intraday_data = load_intraday_data()

    # Module 1: Leverage effect tests
    print("\n[1/5] Leverage effect analysis...")
    lev_results = run_leverage_analysis()

    # Module 2: Asymmetric GARCH (GJR + EGARCH)
    print("\n[2/5] Asymmetric GARCH models...")
    garch_results = run_asymmetric_garch()

    # Module 3: Realized moments from intraday data
    print("\n[3/5] Realized skewness and kurtosis...")
    mom_results = run_realized_moments(intraday_data)

    # Module 4: Asymmetric HAR variants
    print("\n[4/5] Asymmetric HAR models...")
    ahar_results = run_asymmetric_har()

    # Module 5: Plots
    print("\n[5/5] Generating plots...")
    run_all_plots(garch_results)

    elapsed = time.time() - t0
    print(f"\n{'='*55}  DAY 17 COMPLETE  {elapsed:.1f}s  {'='*10}")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:<55} {sz:>8,} bytes")


if __name__ == "__main__":
    main()
