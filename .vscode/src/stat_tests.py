import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

def adf_test(series, max_lag = 10 , regression = "c"):
    series = pd.Series(series).dropna()
    result = adfuller(series , maxlag = max_lag , regression = regression , autolag="AIC")
    stat,pvalue,usedlag,nobs, crit_values , icbest = result 
    return {
        "statistic" : stat, 
        "pvalue" : pvalue,
        "used" : usedlag,
        "nobs" : nobs,
        "crit_values" : crit_values,
        "icbest": icbest,



    }

def ljung_box_test(series, lags = (5,10,20)):
    series = pd.Series(series).dropna()
    lb = acorr_ljungbox(series, lags=list(lags), return_df = True)
    lb.index.name = "lag"
    return lb


def compute_acf_pacf(series , nlags = 40):
    series = pd.Series(series).dropna()
    acf_vals = acf(series , nlags = nlags , fft = True)
    pacf_vals = pacf(series , nlags = nlags , method = "ywm")
    return {"acf" : acf_vals , "pacf" : pacf_vals}

def plot_acf_pacf(series , title_prefix , outdir , nlags = 40):
    outdir = Path(outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    series = pd.Series(series).dropna()
    acf_vals = acf(series, nlags = nlags , fft=True)
    pacf_vals = pacf(series , nlags = nlags , method = "ywm")

    fig,axes = plt.subplots(1,2 , figsize = (10,4))
    lags = np.arange(len(acf_vals))

    axes[0].stem(lags, acf_vals)
    axes[0].axhline(0, color = "black" , linewidth = 0.8)
    axes[0].set_title(f"{title_prefix} ACF")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")

    axes[1].stem(lags, pacf_vals)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{title_prefix} PACF")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("PACF")

    fig.tight_layout()
    out_path = outdir / f"{title_prefix.replace(' ', '_').lower()}_acf_pacf.png"
    fig.savefig(out_path , dpi = 150)
    plt.close(fig)
    return out_path

