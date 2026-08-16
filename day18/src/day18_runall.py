import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which cannot encode the unicode glyphs
# the day18 modules print; force UTF-8 so the driver runs on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from day18_granger_causality import run_granger_causality
from day18_var_model import run_var_model
from day18_connectedness import run_connectedness
from day18_spillover_forecasting import run_spillover_forecasting
from day18_plots import run_all_plots


def main():
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  DAY 18 — Volatility Spillovers & Connectedness")
    print(f"{'='*55}")

    print("\n [1/5] Granger Causality Tests....")
    granger_results = run_granger_causality()

    print("\n [2/5] VAR model ( coeffecients , IRF , FEVD)....")
    var_results = run_var_model()

    print("\n [3/5] Diebold-Yilmaz connectedness....")
    connect_results = run_connectedness(var_results)

    print("\n [4/5] Spillover-augmented forecasting....")
    spill_results = run_spillover_forecasting(connect_results)

    print("\n [5/5] Generating Plots....")
    run_all_plots()

    elapsed = time.time() - t0
    print(f"\n{'='*55} DAY 18 COMPLETE {elapsed:.1f}s {'='*10}")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f" {f:<55} {sz:>8,} bytes")


if __name__ == "__main__":
    main()
