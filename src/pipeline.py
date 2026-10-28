"""End-to-end research pipeline. Call `run(synthetic=False)`.

Order: data -> EWMA sigma -> labels -> features -> HAR vol (walk-forward) -> classifiers
(purged walk-forward, quarterly refits) -> signals -> backtests -> robustness -> trade cards.
Design choices (features, thresholds, barriers, models, selection rule) were fixed before
the test year was evaluated. The test year is also walk-forward (quarterly refits on past
data only), which is how the tool would actually run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

from . import backtest as bt
from .card import EDUCATION, headline, hysteresis, reasons, tier_of
from .data import load_binance_csv, load_market, make_synthetic
from .features import FEATURES, build_features
from .labels import triple_barrier
from .models import CLASSES, CLS_MAP, EMBARGO, quarterly_folds, walk_forward
from .scorecard import walk_forward_score
from .vol import HAR, ewma_sigma_24h, har_components, qlike

TP, SL, HZN = 1.5, 1.0, 6
WF_START = "2022-01-01"
COST_RT = 2 * (bt.FEE + bt.SLIP)
PCOLS = [f"p_{c}" for c in CLASSES]


# ----------------------------------------------------------------------------- helpers
def clf_metrics(d: pd.DataFrame, pr: pd.DataFrame) -> dict:
    m = d["outcome"].notna() & pr["p_target"].notna()
    d, pr = d[m], pr[m]
    y = d["outcome"].map(CLS_MAP).to_numpy()
    P = np.clip(pr[PCOLS].to_numpy(), 1e-6, 1)
    P = P / P.sum(axis=1, keepdims=True)
    tgt = (y == 2).astype(int)
    onehot = np.eye(3)[y]
    # IC on GROSS expected vs GROSS realised result: the cost term (cost / sigma) is known in advance and
    # would otherwise appear on both sides and inflate the correlation mechanically.
    er = sum(pr[f"p_{c}"] * pr[f"R_{c}"] for c in CLASSES)
    realised = d["gross_ret"] / (SL * d["sigma"])
    return {"n": int(m.sum()),
            "rate_stop": float((y == 0).mean()), "rate_timeout": float((y == 1).mean()), "rate_target": float(tgt.mean()),
            "auc_target": float(roc_auc_score(tgt, P[:, 2])), "auc_stop": float(roc_auc_score((y == 0).astype(int), P[:, 0])),
            "logloss": float(log_loss(y, P, labels=[0, 1, 2])),
            "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
            "ic_expectedR": float(spearmanr(er, realised).statistic)}


def expected_R(pr: pd.DataFrame, sigma: pd.Series, sl_mult: float) -> pd.Series:
    """Expected net result per trade, in units of risk: sum_k p_k * R_k  -  costs / risk."""
    gross = sum(pr[f"p_{c}"] * pr[f"R_{c}"] for c in CLASSES)
    return gross - COST_RT / (sl_mult * sigma)


def make_signals(pr, sigma, regime, sl_mult=SL):
    er = expected_R(pr, sigma, sl_mult)
    on = hysteresis(er)
    tiers = er.apply(lambda e: tier_of(e) if np.isfinite(e) else "None")
    trade = on & (regime != "stormy")
    return pd.DataFrame({"expR": er, "state_on": on, "tier": tiers, "regime": regime, "trade": trade,
                         **{c: pr[c] for c in pr.columns}})


def daily_sharpe(r):
    d = (1 + r).groupby(r.index.floor("D")).prod() - 1
    return float(d.mean() / d.std()) if d.std() > 0 else 0.0


def har_walk_forward(comp, folds, times):
    """Out-of-sample HAR sigma + regime (calm / normal / stormy from training percentiles)."""
    sig = pd.Series(np.nan, index=comp.index)
    regime = pd.Series("normal", index=comp.index, dtype=object)
    for a, b in folds:
        te = ((times >= a) & (times < b)).to_numpy()
        if not te.any():
            continue
        first = comp.index[te][0]
        tr = comp.index < first - HZN - EMBARGO
        har = HAR().fit(comp[tr])
        s_tr = har.predict_sigma(comp[tr].dropna(subset=["l_d", "l_w", "l_m"]))
        q30, q90 = np.nanpercentile(s_tr, 30), np.nanpercentile(s_tr, 90)
        s_te = har.predict_sigma(comp[te])
        sig[te] = s_te.values
        regime[te] = np.where(s_te > q90, "stormy", np.where(s_te < q30, "calm", "normal"))
    return sig, regime


# ----------------------------------------------------------------------------- main
def run(synthetic=False, refresh=False, verbose=True, control_edge=0.0, csv=None):
    log = print if verbose else (lambda *a, **k: None)
    data_report = {}
    if csv:
        kl, funding = load_binance_csv(csv, data_report), None
    else:
        kl, funding = make_synthetic(edge=control_edge) if synthetic else load_market(refresh=refresh)
    df = kl.reset_index(drop=True)
    times = df["open_time"]
    log(f"[data] {len(df)} bars {times.iat[0]} -> {times.iat[-1]}  synthetic={synthetic}")

    sigma = ewma_sigma_24h(df["close"])
    lab = triple_barrier(df, sigma, TP, SL, HZN)
    span = df["open_time"].shift(-HZN) - df["open_time"]
    gap = span > pd.Timedelta(hours=4 * HZN)          # a missing candle inside the trade window
    lab.loc[gap.to_numpy(), ["label", "gross_ret", "exit_idx", "outcome"]] = np.nan
    data_report["labels_dropped_for_gaps"] = int(gap.sum())
    feats = build_features(df, funding)
    fcols = [f for f in FEATURES if f in feats.columns and feats[f].notna().mean() > 0.5]
    comp = har_components(df["close"])

    data = pd.concat([df[["open_time"]], lab, feats, sigma.rename("sigma")], axis=1)
    data["sl_mult"] = SL
    data = data[feats[["dist_hi_30d", "vol_z"]].notna().all(axis=1) & sigma.notna()]

    end = times.iat[-1] + pd.Timedelta(hours=4)
    test_start = (end - pd.Timedelta(days=365)).floor("D")
    folds = quarterly_folds(WF_START, end, test_start)
    dev = (data["open_time"] >= WF_START) & (data["open_time"] < test_start)
    tst = data["open_time"] >= test_start
    periods = {"dev": (pd.Timestamp(WF_START, tz="UTC"), test_start), "test": (test_start, end)}
    R = {"meta": {"synthetic": synthetic, "control_edge": control_edge, "first_bar": str(times.iat[0]), "last_bar": str(times.iat[-1]),
                  "wf_start": WF_START, "test_start": str(test_start), "n_folds": len(folds),
                  "features": fcols, "tp_mult": TP, "sl_mult": SL, "horizon_bars": HZN,
                  "cost_round_trip": COST_RT, "funding_available": funding is not None,
                  "data_source": str(csv) if csv else ("synthetic" if synthetic else "binance api"), "data_report": data_report}}

    # ---- volatility --------------------------------------------------------------
    har_sig, regime = har_walk_forward(comp, folds, times)
    regime_d = regime.reindex(data.index)
    rv_fwd = np.exp(comp["l_fwd"])
    R["vol"] = {}
    for name, m in (("dev", dev), ("test", tst)):
        ix = data.index[m]
        R["vol"][name] = {"qlike_har": qlike(rv_fwd[ix].values, (har_sig[ix] ** 2).values),
                          "qlike_ewma": qlike(rv_fwd[ix].values, (sigma[ix] ** 2).values),
                          "rmse_har": float(np.sqrt(np.nanmean((np.sqrt(rv_fwd[ix]) - har_sig[ix]) ** 2))),
                          "rmse_ewma": float(np.sqrt(np.nanmean((np.sqrt(rv_fwd[ix]) - sigma[ix]) ** 2)))}
    log(f"[vol] {R['vol']}")

    # ---- classifiers -------------------------------------------------------------
    preds = {k: walk_forward(data, fcols, k, folds) for k in ("base", "logreg", "gbm")}
    ps, fired, score_models = walk_forward_score(data, fcols, folds)
    preds["score"] = ps[PCOLS + [f"R_{c}" for c in CLASSES]]
    R["clf"] = {k: {"dev": clf_metrics(data[dev], preds[k][dev]), "test": clf_metrics(data[tst], preds[k][tst])}
                for k in preds}
    log("[clf] " + str({k: (round(v["dev"]["auc_target"], 3), round(v["dev"]["logloss"], 4)) for k, v in R["clf"].items()}))

    # ---- signals + backtests -----------------------------------------------------
    always = pd.Series(True, index=data.index) & (regime_d != "stormy")
    bt_res, sig_tables, trials = {}, {}, []
    for kind in ("score", "logreg", "gbm"):
        sg = make_signals(preds[kind], data["sigma"], regime_d)
        sig_tables[kind] = sg
        for variant, s in (("all_signals", sg["trade"]), ("medium_plus", sg["trade"] & sg["tier"].isin(["Medium", "High"]))):
            for per, (a, b) in periods.items():
                r, tr = bt.simulate(df, lab, s, a, b)
                bt_res[f"{kind}|{variant}|{per}"] = bt.metrics(r, tr)
                if per == "dev":
                    trials.append(daily_sharpe(r))
    for per, (a, b) in periods.items():
        r, tr = bt.simulate(df, lab, always, a, b)
        bt_res[f"always_take|-|{per}"] = bt.metrics(r, tr)
        bt_res[f"buy_hold|-|{per}"] = bt.metrics(bt.buy_and_hold(df, a, b))
    R["backtests"] = bt_res

    # pre-registered rule: GBM replaces logreg only if it wins on BOTH dev log-loss (by >0.5%) and dev net Sharpe
    # Pre-registered "simplest model that is not beaten" rule (dev data only):
    # start with the 5-rule points scorecard; a more complex model replaces it only if it beats it on
    # BOTH dev rank-IC of expected result (by > 0.02) AND dev net Sharpe.
    chosen = "score"
    for k in ("logreg", "gbm"):
        if (R["clf"][k]["dev"]["ic_expectedR"] > R["clf"][chosen]["dev"]["ic_expectedR"] + 0.02
                and bt_res[f"{k}|all_signals|dev"]["sharpe"] > max(bt_res[f"{chosen}|all_signals|dev"]["sharpe"], 0)):
            chosen = k
    R["chosen_model"] = chosen
    R["simplicity_test"] = {k: {"n_inputs": {"score": f"<= 5 yes/no rules", "logreg": f"{len(fcols)} features", "gbm": f"{len(fcols)} features, trees"}[k],
                                "dev_auc": R["clf"][k]["dev"]["auc_target"], "test_auc": R["clf"][k]["test"]["auc_target"],
                                "dev_ic": R["clf"][k]["dev"]["ic_expectedR"], "test_ic": R["clf"][k]["test"]["ic_expectedR"],
                                "dev_sharpe": bt_res[f"{k}|all_signals|dev"]["sharpe"], "test_sharpe": bt_res[f"{k}|all_signals|test"]["sharpe"]}
                            for k in ("score", "logreg", "gbm")}
    R["scorecard_latest"] = score_models[-1][2].describe() if score_models else []
    sg = sig_tables[chosen]
    # Kill switch (decided on dev only): the tool may say "buy" only if the dev walk-forward edge is
    # positive after costs AND beats "always take the trade". Otherwise it shows risk context only.
    dv, al = bt_res[f"{chosen}|all_signals|dev"], bt_res["always_take|-|dev"]
    R["edge_validated"] = bool(dv["n_trades"] >= 30 and dv["avg_R"] > 0 and dv["sharpe"] > 0 and dv["sharpe"] > al["sharpe"])

    R["cost_sensitivity_test"] = {}
    for cm in (0.0, 1.0, 2.0):
        r, tr = bt.simulate(df, lab, sg["trade"], test_start, end, cost_mult=cm)
        r2, tr2 = bt.simulate(df, lab, always, test_start, end, cost_mult=cm)
        R["cost_sensitivity_test"][str(cm)] = {"model": bt.metrics(r, tr), "always_take": bt.metrics(r2, tr2)}

    # ---- robustness --------------------------------------------------------------
    rob = {}
    for mult in (0.75, 1.25):
        lab_m = triple_barrier(df, sigma, TP * mult, SL * mult, HZN)
        dm = data.copy()
        dm[lab_m.columns] = lab_m.loc[dm.index]
        dm["sl_mult"] = SL * mult
        pm = walk_forward_score(dm, fcols, folds)[0] if chosen == "score" else walk_forward(dm, fcols, chosen, folds)
        s = make_signals(pm, dm["sigma"], regime_d, SL * mult)["trade"]
        al = pd.Series(True, index=dm.index) & (regime_d != "stormy")
        for per, (a, b) in periods.items():
            r, tr = bt.simulate(df, lab_m, s, a, b)
            r2, tr2 = bt.simulate(df, lab_m, al, a, b)
            mm = dev if per == "dev" else tst
            rob[f"barriers_x{mult}|{per}"] = {"clf": clf_metrics(dm[mm], pm[mm]), "model": bt.metrics(r, tr),
                                              "always_take": bt.metrics(r2, tr2)}
            if per == "dev":
                trials.append(daily_sharpe(r))

    # importance: target-vs-stop coefficient of a logreg fit on all dev data
    _, mods = walk_forward(data, fcols, "logreg", [(periods["test"][0], periods["test"][1])], keep_models=True)
    lr = mods[0][2].model[-1]
    tv = lr.coef_[CLS_MAP["target"]] - lr.coef_[CLS_MAP["stop"]]
    R["coef_target_vs_stop_dev"] = dict(sorted(zip(fcols, np.round(tv, 4).tolist()), key=lambda x: -abs(x[1])))
    top = next(iter(R["coef_target_vs_stop_dev"]))
    keep = [f for f in fcols if f != top]
    p_drop = walk_forward_score(data, keep, folds)[0] if chosen == "score" else walk_forward(data, keep, chosen, folds)
    rob["drop_top_feature"] = {"feature": top, "dev": clf_metrics(data[dev], p_drop[dev]),
                               "test": clf_metrics(data[tst], p_drop[tst])}

    pc = preds[chosen]
    yrs = data["open_time"].dt.year
    by = {}
    for y in sorted(yrs[data["open_time"] >= WF_START].unique()):
        mm = (yrs == y) & data["outcome"].notna() & pc["p_target"].notna()
        if mm.sum() > 200:
            st = mm & sg["trade"]
            by[str(y)] = {**clf_metrics(data[mm], pc[mm]), "coverage": float(sg.loc[mm, "trade"].mean()),
                          "signal_avg_net_R": float(((data.loc[st, "gross_ret"] - COST_RT) / (SL * data.loc[st, "sigma"])).mean()) if st.any() else None}
    rob["by_year"] = by
    rob["by_regime"] = {}
    for g in ("calm", "normal", "stormy"):
        mm = (regime_d == g) & data["outcome"].notna() & pc["p_target"].notna() & (data["open_time"] >= WF_START)
        if mm.sum() > 200:
            rob["by_regime"][g] = clf_metrics(data[mm], pc[mm])
    R["robustness"] = rob

    r, _ = bt.simulate(df, lab, sg["trade"], *periods["dev"])
    d = (1 + r).groupby(r.index.floor("D")).prod() - 1
    R["deflated_sharpe_dev"] = bt.deflated_sharpe(d, trials)

    # ---- honest track record on the test year --------------------------------------
    t_rows = tst & data["outcome"].notna()
    trk = {}
    for tn in ("Low", "Medium", "High"):
        mm = t_rows & sg["trade"] & (sg["tier"] == tn)
        Rr = (data.loc[mm, "gross_ret"] - COST_RT) / (SL * data.loc[mm, "sigma"])
        trk[tn] = {"n_bars": int(mm.sum()), "target_rate": float((data.loc[mm, "outcome"] == "target").mean()) if mm.any() else None,
                   "stop_rate": float((data.loc[mm, "outcome"] == "stop").mean()) if mm.any() else None,
                   "avg_net_R": float(Rr.mean()) if mm.any() else None,
                   "avg_predicted_R": float(sg.loc[mm, "expR"].mean()) if mm.any() else None}
    trk["no_signal_share"] = float(1 - sg.loc[tst, "trade"].mean())
    R["track_record_test"] = trk

    cards = build_cards(df, data, fcols, sg, har_sig, lab, folds, test_start, pc,
                        fired if chosen == "score" else None, ps["score"] if chosen == "score" else None)
    if not R["edge_validated"]:
        for c in cards:
            if c["trade"]:
                c["headline"] = "Model leans up, but its edge is not proven — treat as no signal"
    R["latest_card"] = cards[-1] if cards else None
    trades_test = bt.simulate(df, lab, sg["trade"], test_start, end)[1]
    return {"results": R, "df": df, "data": data, "signals": sg, "preds": preds, "har_sigma": har_sig,
            "regime": regime, "labels": lab, "cards": cards, "trades_test": trades_test,
            "test_start": test_start, "end": end, "periods": periods, "always": always}


def _dollars(d_stop, d_target, equity=1000.0):
    """Card arithmetic for a $1,000 account with the same sizing as the sized backtest."""
    w = float(bt.risk_weight(d_stop))
    n = equity * w
    c = bt.FEE + bt.SLIP
    return {"size_for_1pct_risk_on_1000": round(n, 2),
            "loss_if_stop": round(n * (d_stop + 2 * c), 2),
            "gain_if_target": round(n * (d_target - 2 * c), 2)}


def out_of_10(p):
    """Largest-remainder rounding of probabilities into 10 icons (natural frequencies)."""
    raw = np.asarray(p) * 10
    n = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - n))[: 10 - n.sum()]:
        n[k] += 1
    return n.tolist()


def build_cards(df, data, fcols, sg, har_sig, lab, folds, test_start, pc, fired=None, score=None):
    """One card per test-year bar. Reasons come from the logreg in force at that time
    (interpretable even if GBM were chosen); 'similar setups' search that fold's training history."""
    cards = []
    test_folds = [(a, b) for a, b in folds if a >= test_start]
    _, mods = walk_forward(data, fcols, "logreg", test_folds, keep_models=True)
    for a, b, m, tr in mods:
        te = ((data["open_time"] >= a) & (data["open_time"] < b)).to_numpy()
        imp, sc, lr = m.model
        Z_te = sc.transform(imp.transform(data.loc[te, fcols].to_numpy()))
        tv = lr.coef_[CLS_MAP["target"]] - lr.coef_[CLS_MAP["stop"]]
        for j, i in enumerate(data.index[te]):
            s = sg.loc[i]
            z = Z_te[j]
            missing = bool(data.loc[i, fcols].isna().any())
            close, sig = float(df.at[i, "close"]), float(data.at[i, "sigma"])
            hs = float(har_sig.get(i, np.nan))
            pT, pS, pE = float(pc.at[i, "p_target"]), float(pc.at[i, "p_stop"]), float(pc.at[i, "p_expiry"])
            RT, RS = float(pc.at[i, "R_target"]), float(pc.at[i, "R_stop"])
            c_R = COST_RT / (SL * sig)
            q_need = (abs(RS) + c_R) / (RT + abs(RS))       # win share needed among resolved trades
            outc = lab.at[i, "outcome"]
            cards.append({
                "time": str(data.at[i, "open_time"] + pd.Timedelta(hours=4)),
                "price": round(close, 2),
                "headline": "Data unavailable \u2014 no signal" if missing else headline(bool(s["state_on"]), s["tier"], s["regime"]),
                "data_ok": not missing,
                "confidence": (s["tier"] if s["tier"] != "None" else "Low") if s["trade"] else "None",
                "trade": bool(s["trade"]), "regime": s["regime"],
                "p_target": round(pT, 3), "p_stop": round(pS, 3), "p_timeout": round(pE, 3),
                "win_share": round(pT / (pT + pS), 3) if pT + pS > 0 else None, "win_share_needed": round(q_need, 3),
                "expected_R": round(float(s["expR"]), 3),
                "entry": round(close, 2), "stop": round(close * (1 - SL * sig), 2), "target": round(close * (1 + TP * sig), 2),
                "stop_pct": round(100 * SL * sig, 2), "target_pct": round(100 * TP * sig, 2),
                **_dollars(SL * sig, TP * sig),
                "expires": str(data.at[i, "open_time"] + pd.Timedelta(hours=8))[:16],
                "expected_range_pct": round(100 * hs, 2) if np.isfinite(hs) else None,
                "reasons": ([{"text": r["text"], "points": r["points"],
                              "direction": "supports buying" if r["points"] > 0 else "argues against buying"}
                             for r in sorted(fired.get(i, []), key=lambda r: -abs(r["points"]))]
                            if fired is not None else reasons(dict(zip(fcols, tv * z)), dict(zip(fcols, z)))),
                "score": int(score[i]) if score is not None and np.isfinite(score[i]) else None,
                "of10": out_of_10([pT, pE, pS]),
                "dollars_per_10_risked": round(10 * float(s["expR"]), 2),
                "education": EDUCATION[len(cards) % len(EDUCATION)],
                "outcome": outc if isinstance(outc, str) else "pending",
            })
    return cards
