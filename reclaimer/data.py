"""Loading the usage history and turning it into forecasting features."""
from __future__ import annotations
import numpy as np
import pandas as pd

SCRAPE_S = 5      # how often the CSV was sampled (kubectl top loop)
WINDOW_S = 15     # metrics-server resolution (default --metric-resolution=15s)


def load_usage_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.rename(columns={"Timestamp": "ts", "CPU_millicores": "cpu", "Memory_Mi": "mem"})
    return df.sort_values("ts").reset_index(drop=True)


def collapse_to_readings(df: pd.DataFrame, scrape_s: int = SCRAPE_S, window_s: int = WINDOW_S) -> pd.DataFrame:
    """Collapse the 5-second samples into one value per metrics-server window.

    `kubectl top` / the Metrics API only refreshes every ~15 s, so polling every 5 s
    returns each reading 3 times (a staircase).  Keeping those repeats would make
    "next value == current value" true 2 times out of 3 and inflate any forecast
    accuracy.  A run of identical (cpu, mem) samples lasting `d` seconds is therefore
    turned into round(d / 15) readings (>= 1) - which also preserves genuine
    back-to-back identical readings (e.g. two consecutive 0 m windows).
    """
    key = (df[["cpu", "mem"]] != df[["cpu", "mem"]].shift()).any(axis=1).cumsum()
    runs = df.groupby(key).agg(ts=("ts", "first"), end=("ts", "last"),
                               cpu=("cpu", "first"), mem=("mem", "first"))
    dur = (runs["end"] - runs["ts"]).dt.total_seconds() + scrape_s
    n = (dur / window_s).round().clip(lower=1).astype(int)
    out = runs.loc[runs.index.repeat(n.values), ["ts", "cpu", "mem"]].reset_index(drop=True)
    # re-stamp on a regular grid inside each run so the index is a clean 15 s series
    offs = np.concatenate([np.arange(k) for k in n.values]) * window_s
    out["ts"] = out["ts"] + pd.to_timedelta(offs, unit="s")
    return out


# ---------------------------------------------------------------- features
FEATURE_NAMES = None  # filled lazily by feature_names()


def feature_names(n_lags: int) -> list[str]:
    names = [f"lag{k}" for k in range(n_lags)]
    names += ["mean4", "max4", "min4", "std4", "mean_all", "max_all", "std_all",
              "zero_frac", "diff1", "diff2"]
    return names


def features_at(hist: np.ndarray, n_lags: int) -> np.ndarray:
    """Feature vector computed from the last `n_lags` readings of `hist` only."""
    w = np.asarray(hist[-n_lags:], dtype=float)
    if len(w) < n_lags:                       # left-pad with the oldest value
        w = np.concatenate([np.full(n_lags - len(w), w[0]), w])
    lags = w[::-1]                            # lag0 = most recent
    w4 = w[-4:]
    f = list(lags)
    f += [w4.mean(), w4.max(), w4.min(), w4.std(), w.mean(), w.max(), w.std(),
          float((w <= 0.01 * max(w.max(), 1e-9)).mean()),
          w[-1] - w[-2], w[-2] - w[-3]]
    return np.array(f, dtype=float)


def build_dataset(series: np.ndarray, n_lags: int, horizon: int):
    """Supervised windows: X_i from series[:i+1], y_i = max(series[i+1 : i+1+horizon]).

    Only windows whose target is fully observed are returned, so a model fitted at
    time t never sees information from beyond t.
    """
    X, y, idx = [], [], []
    for i in range(n_lags - 1, len(series) - horizon):
        X.append(features_at(series[: i + 1], n_lags))
        y.append(series[i + 1: i + 1 + horizon].max())
        idx.append(i)
    return np.array(X), np.array(y), np.array(idx)
