"""Data access: Binance BTCUSDT 4h klines + perpetual funding rates.

Everything is cached to ./data so the pipeline runs offline after the first download.
`make_synthetic` builds a placebo market (GARCH random walk, no directional edge)
used to check that the pipeline does NOT find an edge where none exists.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BAR = pd.Timedelta(hours=4)

# data-api.binance.vision serves public market data only and is reachable from more regions
SPOT_URLS = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def download_klines(symbol="BTCUSDT", interval="4h", start="2019-09-01", end=None) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    last_err = None
    for base in SPOT_URLS:
        try:
            rows, cur = [], start_ms
            while cur < end_ms:
                batch = _get_json(f"{base}?symbol={symbol}&interval={interval}&startTime={cur}&limit=1000")
                if not batch:
                    break
                rows += batch
                cur = batch[-1][0] + 1
                time.sleep(0.15)
            break
        except Exception as e:  # try next mirror
            last_err = e
            rows = []
    if not rows:
        raise RuntimeError(f"Could not download klines: {last_err}")
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"]]
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    # drop the still-forming last bar
    df = df[df["open_time"] + BAR <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)
    return df


def download_funding(symbol="BTCUSDT", start="2019-09-01") -> pd.DataFrame:
    cur = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        batch = _get_json(f"{FUNDING_URL}?symbol={symbol}&startTime={cur}&limit=1000")
        if not batch:
            break
        rows += batch
        cur = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["funding_time", "funding_rate"]].sort_values("funding_time").reset_index(drop=True)


def load_binance_csv(path, report=None):
    """Load a Binance kline CSV export (columns 'Open time', 'Open', ..., 'Taker buy base asset volume')
    and clean it. Cleaning found on the 2018-2026 export: two corrupted duplicate 4h rows on 2018-07-06
    (prices near $108,000 when BTC traded near $6,500). For duplicated timestamps we keep the row closest
    to its neighbours; any remaining bar more than 25% away from the local median is dropped and reported."""
    d = pd.read_csv(path)
    d = d[d["Open time"].notna() & d["Close"].notna()]          # the export ends with an empty/partial row
    d["open_time"] = pd.to_datetime(d["Open time"].astype(str).str.replace(" UTC", "").str.strip(), utc=True)
    d = d.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
                          "Taker buy base asset volume": "taker_buy_base"})
    d = d[["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"]].sort_values("open_time").reset_index(drop=True)
    med = d["close"].rolling(13, center=True, min_periods=3).median()
    dev = (d["close"] / med - 1).abs()
    n_raw, n_dup = len(d), int(d["open_time"].duplicated(keep=False).sum())
    d = d.assign(_dev=dev).sort_values(["open_time", "_dev"]).drop_duplicates("open_time", keep="first")
    outliers = d[d["_dev"] > 0.25]
    d = d[d["_dev"] <= 0.25].drop(columns="_dev").sort_values("open_time").reset_index(drop=True)
    step = d["open_time"].diff().dropna().mode().iat[0]
    full = pd.date_range(d["open_time"].iat[0], d["open_time"].iat[-1], freq=step)
    missing = full.difference(d["open_time"])
    if report is not None:
        report.update({"rows_raw": n_raw, "duplicate_timestamp_rows": n_dup, "outliers_dropped": len(outliers),
                       "missing_bars": len(missing), "missing_examples": [str(x) for x in missing[:5]],
                       "rows_clean": len(d), "first": str(d["open_time"].iat[0]), "last": str(d["open_time"].iat[-1])})
    return d


def load_market(refresh=False, start="2019-09-01"):
    """Return (klines, funding). Funding may be None if the futures endpoint is blocked."""
    DATA_DIR.mkdir(exist_ok=True)
    kp, fp = DATA_DIR / "btcusdt_4h.csv", DATA_DIR / "btcusdt_funding.csv"
    if refresh or not kp.exists():
        download_klines(start=start).to_csv(kp, index=False)
    kl = pd.read_csv(kp, parse_dates=["open_time"])
    funding = None
    try:
        if refresh or not fp.exists():
            download_funding(start=start).to_csv(fp, index=False)
        funding = pd.read_csv(fp, parse_dates=["funding_time"])
    except Exception as e:
        print(f"[data] funding unavailable ({e}); funding features will be dropped")
    return kl, funding


def make_synthetic(start="2019-09-01", end="2026-10-27", seed=7, edge=0.0):
    """Control markets. GARCH(1,1) t-innovations; volatility clusters like BTC.

    edge=0   -> NEGATIVE control (placebo): no directional edge; the pipeline should find nothing.
    edge>0   -> POSITIVE control: next-bar drift = edge * sigma * (standardised 24h taker imbalance),
                a small, known, causal edge; the pipeline should find it.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="4h", tz="UTC", inclusive="left")
    n = len(idx)
    tk_ratio = np.clip(0.5 + rng.normal(0, 0.03, n), 0.3, 0.7)
    sgn = pd.Series((tk_ratio - 0.5) / 0.03).rolling(6, min_periods=1).mean().to_numpy() * np.sqrt(6)
    omega, alpha, beta = 2.9e-6, 0.08, 0.90  # ~1.2% per 4h bar, like BTC
    h = omega / (1 - alpha - beta)
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    r = np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(h) * (z[i] + (edge * sgn[i - 1] if i else 0.0))
        h = omega + alpha * r[i] ** 2 + beta * h
    close = 10_000 * np.exp(np.cumsum(r))
    open_ = np.r_[10_000, close[:-1]]
    sig = np.abs(r) + 0.004
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.5, n)) * sig)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.5, n)) * sig)
    vol = np.exp(8 + 40 * np.abs(r) + rng.normal(0, 0.3, n))
    taker = vol * tk_ratio
    kl = pd.DataFrame({"open_time": idx, "open": open_, "high": high, "low": low, "close": close,
                       "volume": vol, "taker_buy_base": taker})
    ft = pd.date_range(start, end, freq="8h", tz="UTC", inclusive="left")
    fr = 1e-4 + 1e-4 * rng.standard_normal(len(ft))
    funding = pd.DataFrame({"funding_time": ft, "funding_rate": fr})
    return kl, funding
