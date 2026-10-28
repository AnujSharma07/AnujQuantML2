"""Extra 'is it luck?' checks run on the real data. Nothing here changes the strategy; it only measures it.

1. Random-entry benchmark: same number of trades, same stop/target/24h rules, entries at random times
   (non-overlapping). How often does chance do as well as the model?  -> one-sided p-value.
2. IC significance by circular-shift test: shift the model's scores in time by a random offset (at least
   30 days) and recompute the rank-IC. This keeps autocorrelation in both series but breaks their timing.
3. Profit attribution by calendar year.
4. Label quality: how often one 4h candle touched both stop and target (counted as a stop), and how many
   of those would flip if resolved with 1h candles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import backtest as bt


def _daily_sharpe(r):
    d = (1 + r).groupby(r.index.floor("D")).prod() - 1
    return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else 0.0


def random_entry_benchmark(o, period="dev", n_sims=500, seed=0):
    df, lab, sg = o["df"], o["labels"], o["signals"]
    a, b = o["periods"][period]
    real_r, real_tr = bt.simulate(df, lab, sg["trade"], a, b)
    n = len(real_tr)
    if n == 0:
        return {"n_trades": 0, "note": "no trades"}
    real_s, real_tot = _daily_sharpe(real_r), float((1 + real_r).prod() - 1)
    rng = np.random.default_rng(seed)
    times = df["open_time"]
    cand = np.where(((times >= a) & (times < b)).to_numpy() & lab["outcome"].notna().to_numpy())[0]
    sims_s, sims_tot = [], []
    for _ in range(n_sims):
        pick = np.sort(rng.choice(cand, size=min(len(cand), n * 3), replace=False))
        sig = pd.Series(False, index=df.index)
        sig.iloc[pick] = True
        r, tr = bt.simulate(df, lab, sig, a, b)
        # keep exactly n trades: truncate the series after the n-th exit
        if len(tr) > n:
            cut = tr["exit_time"].iat[n - 1]
            r = r[r.index <= cut].reindex(r.index, fill_value=0.0)
        sims_s.append(_daily_sharpe(r)); sims_tot.append(float((1 + r).prod() - 1))
    sims_s, sims_tot = np.array(sims_s), np.array(sims_tot)
    return {"n_trades": n, "model_sharpe": real_s, "model_total_return": real_tot,
            "random_sharpe_median": float(np.median(sims_s)), "random_total_return_median": float(np.median(sims_tot)),
            "p_value_sharpe": float((sims_s >= real_s).mean()), "p_value_return": float((sims_tot >= real_tot).mean()),
            "n_sims": n_sims}


def ic_shift_test(o, period="dev", n_sims=500, seed=0, min_shift_days=30):
    data, sg = o["data"], o["signals"]
    a, b = o["periods"][period]
    m = (data["open_time"] >= a) & (data["open_time"] < b) & data["outcome"].notna() & sg["expR"].notna()
    # gross on both sides (see clf_metrics): the known cost/sigma term must not be shared
    x = (sg.loc[m, "expR"] + 2 * (bt.FEE + bt.SLIP) / (data.loc[m, "sl_mult"] * data.loc[m, "sigma"])).to_numpy()
    y = (data.loc[m, "gross_ret"] / (data.loc[m, "sl_mult"] * data.loc[m, "sigma"])).to_numpy()
    ic = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    k0 = min_shift_days * 6
    sims = [float(spearmanr(np.roll(x, rng.integers(k0, len(x) - k0)), y).statistic) for _ in range(n_sims)]
    return {"ic": ic, "p_value": float((np.array(sims) >= ic).mean()), "null_95pct": float(np.percentile(sims, 95)), "n": int(m.sum())}


def by_year(o):
    r, tr = bt.simulate(o["df"], o["labels"], o["signals"]["trade"], o["periods"]["dev"][0], o["end"])
    if not len(tr):
        return {}
    tr["year"] = pd.to_datetime(tr["entry_time"]).dt.year
    g = tr.groupby("year")
    return {str(y): {"trades": int(len(t)), "net_return": float((1 + t["net"]).prod() - 1),
                     "target": int((t["outcome"] == "target").sum()), "stop": int((t["outcome"] == "stop").sum()),
                     "timeout": int((t["outcome"] == "expiry").sum())} for y, t in g}


def same_bar_check(o, path_1h):
    """Fraction of labels where one 4h candle touched both barriers, and how 1h candles would resolve them."""
    from .data import load_binance_csv
    df, lab = o["df"], o["labels"]
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    amb = []
    for t in np.where(lab["outcome"].to_numpy() == "stop")[0]:
        k = int(lab["exit_idx"].iat[t])
        if h[k] >= lab["tp_px"].iat[t] and l[k] <= lab["sl_px"].iat[t]:
            amb.append((t, k))
    out = {"labels": int(lab["outcome"].notna().sum()), "ambiguous": len(amb)}
    out["ambiguous_share"] = out["ambiguous"] / max(out["labels"], 1)
    try:
        h1 = load_binance_csv(path_1h).set_index("open_time")
        flips = 0
        for t, k in amb:
            start = df["open_time"].iat[k]
            sub = h1.loc[start:start + pd.Timedelta(hours=3)]
            for _, row in sub.iterrows():
                hit_sl, hit_tp = row["low"] <= lab["sl_px"].iat[t], row["high"] >= lab["tp_px"].iat[t]
                if hit_sl:
                    break
                if hit_tp:
                    flips += 1
                    break
        out["would_flip_to_target_with_1h"] = flips
    except Exception as e:   # 1h file optional
        out["note"] = f"1h check skipped: {e}"
    return out


def run_all_checks(o, path_1h=None):
    res = {"random_entry_dev": random_entry_benchmark(o, "dev"),
           "random_entry_test": random_entry_benchmark(o, "test"),
           "ic_shift_dev": ic_shift_test(o, "dev"), "ic_shift_test": ic_shift_test(o, "test"),
           "by_year_trades": by_year(o)}
    if path_1h:
        res["same_bar"] = same_bar_check(o, path_1h)
    res["release_gate"] = release_gate(o)
    res["sized_short_term"] = sized_results(o)
    return res


def release_gate(o, n_boot=5000, seed=0):
    """Stricter release gate for the short-term buy signal (from an independent review of the plan).
    All four must hold on the development period, or the card stays in research mode:
      1. at least 100 trades;
      2. average net R > 0 with a weekly block-bootstrap 95% lower bound above zero;
      3. average net R still > 0 at twice the assumed costs;
      4. net Sharpe better than taking every trade with the same rules.
    """
    df, lab, sg = o["df"], o["labels"], o["signals"]
    a, b = o["periods"]["dev"]
    _, tr = bt.simulate(df, lab, sg["trade"], a, b)
    _, tr2 = bt.simulate(df, lab, sg["trade"], a, b, cost_mult=2.0)
    r_m, _ = bt.simulate(df, lab, sg["trade"], a, b)
    r_a, _ = bt.simulate(df, lab, o["always"], a, b)
    if len(tr) == 0:
        return {"passed": False, "n_trades": 0}
    R = (tr["net"] / tr["sigma"]).to_numpy()
    wk = pd.to_datetime(tr["entry_time"]).dt.to_period("W").astype(str).to_numpy()
    weeks = pd.unique(wk)
    groups = [R[wk == w] for w in weeks]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    lb = float(np.percentile(means, 2.5))
    R2 = float((tr2["net"] / tr2["sigma"]).mean())
    res = {"n_trades": int(len(tr)), "avg_net_R": float(R.mean()), "bootstrap_lower_95": lb,
           "upper_95": float(np.percentile(means, 97.5)), "avg_net_R_double_costs": R2,
           "sharpe_model": _daily_sharpe(r_m), "sharpe_take_every_trade": _daily_sharpe(r_a)}
    res["checks"] = {"at_least_100_trades": res["n_trades"] >= 100, "lower_bound_above_zero": lb > 0,
                     "positive_at_double_costs": R2 > 0, "beats_take_every_trade": res["sharpe_model"] > res["sharpe_take_every_trade"]}
    res["passed"] = all(res["checks"].values())
    return res


def sized_results(o):
    """The short-term strategy with the card's 1%-risk sizing instead of the whole account."""
    out = {}
    for per, (a, b) in o["periods"].items():
        r, tr = bt.simulate(o["df"], o["labels"], o["signals"]["trade"], a, b, sized=True)
        m = bt.metrics(r, tr) if len(tr) else {"n_trades": 0}
        if len(tr):
            m["avg_weight_in_trade"] = float(tr["weight"].mean())
        out[per] = m
    return out
