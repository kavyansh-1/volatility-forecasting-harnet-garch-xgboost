# ─────────────────────────────────────────────────────────────
# day15_monitoring.py
# Model monitoring system for deployed volatility forecasters.
#
# WHAT DOES A MODEL MONITORING SYSTEM DO?
# ─────────────────────────────────────────────────────────────
# In production, a model generates predictions every day.
# The monitoring system answers ONE question continuously:
#
#   "Is this model still working as well as it did during
#    validation, or has something broken?"
#
# It tracks three signal types:
#
#   1. PERFORMANCE DEGRADATION
#      Rolling RMSE and QLIKE compared to a historical baseline.
#      If rolling RMSE is consistently above the 2-sigma band
#      of training performance → performance alert.
#
#   2. DISTRIBUTION SHIFT (from Module 2)
#      PSI scores for inputs and target.
#      If PSI exceeds threshold → covariate shift alert.
#
#   3. PREDICTION HEALTH
#      Are predictions staying within reasonable bounds?
#      Check for: all-same-value outputs (model stuck),
#      negative predictions (physically impossible for RV),
#      predictions that are extreme outliers (>10x historical mean).
#
# ALERT LEVELS:
#   GREEN  : all metrics within expected range
#   YELLOW : one signal fired (investigate)
#   RED    : two or more signals fired (likely retraining needed)
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS      = ["SPY", "QQQ", "AAPL"]
ROLL_PERF_WIN = 63     # rolling performance window (1 quarter)
N_SIGMA       = 2.0    # alert threshold: N standard deviations above baseline
PSI_WARN      = 0.10   # PSI moderate drift threshold
PSI_ALERT     = 0.25   # PSI major drift threshold


class ModelMonitor:
    """
    Tracks model performance and health metrics over time.
    Designed to be updated daily in a production setting.

    Maintains a log of:
        - Daily prediction errors (for rolling RMSE)
        - PSI drift scores (from drift_detection module)
        - Prediction health flags
        - Alert history
    """
    def __init__(self,
                 ticker:    str,
                 baseline_rmse:  float,
                 baseline_qlike: float,
                 n_sigma:   float = N_SIGMA):
        self.ticker         = ticker
        self.baseline_rmse  = baseline_rmse
        self.baseline_qlike = baseline_qlike
        self.n_sigma        = n_sigma

        # Estimate sigma of RMSE from historical variability
        # (conservative: use 20% of baseline as sigma)
        self.sigma_rmse  = baseline_rmse  * 0.20
        self.sigma_qlike = baseline_qlike * 0.20

        self.alert_upper_rmse  = baseline_rmse  + n_sigma * self.sigma_rmse
        self.alert_upper_qlike = (baseline_qlike +
                                   n_sigma * self.sigma_qlike)

        self.log = []    # list of daily monitoring records

    def _compute_rmse(self, y, yhat):
        return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat))**2))

    def _compute_qlike(self, y, yhat, floor=1e-8):
        h = np.maximum(yhat, floor)
        v = np.maximum(y,    floor)
        return float(np.mean(np.log(h) + v/h))

    def _check_prediction_health(self, preds: np.ndarray) -> dict:
        """
        Checks for degenerate prediction behaviours.

        all_same: model outputs same value every day (frozen weights)
        has_negative: RV cannot be negative — any negative = error
        extreme_outlier: prediction > 10x the mean (data or model error)
        """
        if len(preds) == 0:
            return {"all_same": 0, "has_negative": 0, "extreme_outlier": 0}

        all_same       = int(np.std(preds) < 1e-8)
        has_negative   = int(np.any(preds < 0))
        mean_pred      = np.abs(preds).mean()
        extreme_outlier= int(np.any(np.abs(preds) > 10 * mean_pred))

        return {
            "all_same"       : all_same,
            "has_negative"   : has_negative,
            "extreme_outlier": extreme_outlier,
        }

    def _classify_alert(self, perf_alert: int,
                         drift_alert:  int,
                         health_alert: int) -> str:
        """
        Combine signal counts into a traffic-light alert level.
        """
        total = perf_alert + drift_alert + health_alert
        if total == 0:
            return "GREEN"
        elif total == 1:
            return "YELLOW"
        else:
            return "RED"

    def log_window(self,
                   date:          pd.Timestamp,
                   y_true:        np.ndarray,
                   y_pred:        np.ndarray,
                   psi_rv:        float = None,
                   psi_feat_max:  float = None) -> dict:
        """
        Log monitoring metrics for one time window.

        Parameters
        ----------
        date         : end date of the monitoring window
        y_true       : actual RV values in the window
        y_pred       : model predictions in the window
        psi_rv       : PSI score for the RV distribution (from drift module)
        psi_feat_max : max PSI score across all input features
        """
        rmse_w  = self._compute_rmse(y_true, y_pred)
        qlike_w = self._compute_qlike(y_true, y_pred)
        health  = self._check_prediction_health(y_pred)

        # Alerts
        perf_alert  = int(rmse_w > self.alert_upper_rmse)
        drift_alert = int((psi_rv is not None and psi_rv >= PSI_WARN) or
                           (psi_feat_max is not None and psi_feat_max >= PSI_WARN))
        health_alert = int(any(health.values()))

        alert_level = self._classify_alert(perf_alert, drift_alert, health_alert)

        record = {
            "date"           : date,
            "ticker"         : self.ticker,
            "rmse"           : round(rmse_w,  8),
            "qlike"          : round(qlike_w, 6),
            "baseline_rmse"  : round(self.baseline_rmse, 8),
            "alert_upper_rmse": round(self.alert_upper_rmse, 8),
            "psi_rv"         : round(psi_rv,        4) if psi_rv else None,
            "psi_feat_max"   : round(psi_feat_max,  4) if psi_feat_max else None,
            "perf_alert"     : perf_alert,
            "drift_alert"    : drift_alert,
            "health_alert"   : health_alert,
            **{f"health_{k}": v for k, v in health.items()},
            "alert_level"    : alert_level,
        }
        self.log.append(record)
        return record

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def alert_rate(self) -> dict:
        df = self.summary()
        return {
            "total_windows"  : len(df),
            "green_pct"      : round((df["alert_level"]=="GREEN").mean()*100, 1),
            "yellow_pct"     : round((df["alert_level"]=="YELLOW").mean()*100, 1),
            "red_pct"        : round((df["alert_level"]=="RED").mean()*100, 1),
            "perf_alerts"    : int(df["perf_alert"].sum()),
            "drift_alerts"   : int(df["drift_alert"].sum()),
            "health_alerts"  : int(df["health_alert"].sum()),
        }


def run_monitoring(online_results: dict,
                    drift_results:  dict) -> dict:
    """
    Build monitoring logs for all tickers using online model predictions
    and drift scores.
    """
    print(f"\n{'='*55}")
    print("  DAY 15 — Model Monitoring")
    print(f"{'='*55}")
    print(f"  Rolling window : {ROLL_PERF_WIN} days")
    print(f"  Alert at       : {N_SIGMA}σ above baseline RMSE")

    all_monitors = {}
    all_logs     = []

    for ticker in TICKERS:
        if ticker not in online_results:
            continue

        res = online_results[ticker]
        y   = res["y"].values
        p   = res["preds_ewma"]

        # Baseline: RMSE on first ROLL_PERF_WIN post-burn-in predictions
        burn  = 252
        valid = ~np.isnan(p)
        if valid.sum() < ROLL_PERF_WIN + 10:
            print(f"  ⚠ {ticker}: insufficient predictions")
            continue

        first_valid = np.where(valid)[0][0]
        baseline_win = slice(first_valid, first_valid + ROLL_PERF_WIN)
        base_rmse  = np.sqrt(np.mean(
            (y[baseline_win] - p[baseline_win])**2
        ))
        base_qlike = float(np.mean(
            np.log(np.maximum(p[baseline_win], 1e-8)) +
            np.maximum(y[baseline_win], 1e-8) /
            np.maximum(p[baseline_win], 1e-8)
        ))

        print(f"\n  {ticker}:")
        print(f"    Baseline RMSE  : {base_rmse:.6f}")
        print(f"    Alert threshold: {base_rmse * (1 + N_SIGMA * 0.20):.6f}")

        monitor = ModelMonitor(ticker, base_rmse, base_qlike)

        # Get drift data for this ticker
        rv_drift_df   = drift_results.get(ticker, {}).get("rv_drift", pd.DataFrame())
        feat_drift_df = drift_results.get(ticker, {}).get("feat_drift", pd.DataFrame())

        # Slide monitoring window over test period
        n    = len(y)
        step = ROLL_PERF_WIN // 3   # monitor 3x per window length

        idx = first_valid + ROLL_PERF_WIN
        while idx + ROLL_PERF_WIN <= n:
            win_slice = slice(idx, idx + ROLL_PERF_WIN)
            y_w = y[win_slice]
            p_w = p[win_slice]

            if np.any(np.isnan(p_w)):
                idx += step
                continue

            date = res["X"].index[idx + ROLL_PERF_WIN - 1]

            # Look up PSI scores near this date
            psi_rv  = None
            psi_max = None
            if not rv_drift_df.empty:
                rv_drift_df["date"] = pd.to_datetime(rv_drift_df["date"])
                near = rv_drift_df[
                    rv_drift_df["date"] <= pd.Timestamp(date)
                ]
                if len(near):
                    psi_rv = float(near["psi"].iloc[-1])

            if not feat_drift_df.empty:
                feat_drift_df["date"] = pd.to_datetime(feat_drift_df["date"])
                psi_cols = [c for c in feat_drift_df.columns if c.startswith("psi_")]
                near_f = feat_drift_df[
                    feat_drift_df["date"] <= pd.Timestamp(date)
                ]
                if len(near_f) and psi_cols:
                    psi_max = float(near_f[psi_cols].iloc[-1].max())

            record = monitor.log_window(date, y_w, p_w, psi_rv, psi_max)
            idx += step

        all_monitors[ticker] = monitor
        log_df = monitor.summary()
        all_logs.append(log_df)

        rates = monitor.alert_rate()
        print(f"    Monitoring windows: {rates['total_windows']}")
        print(f"    GREEN={rates['green_pct']}%  "
              f"YELLOW={rates['yellow_pct']}%  "
              f"RED={rates['red_pct']}%")

    if all_logs:
        combined = pd.concat(all_logs, ignore_index=True)
        combined.to_csv(
            os.path.join(OUT_DIR, "day15_monitoring_log.csv"), index=False
        )
        print(f"\n  ✓ day15_monitoring_log.csv")

    return all_monitors