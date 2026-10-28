"""Figures, auto-generated results appendix (PDF) and the trader dashboard (HTML)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest as bt
from .pipeline import COST_RT, SL

INK, MUTED, ACCENT, GOOD, BAD, GREY = "#1f2328", "#6e7781", "#b7791f", "#1a7f37", "#cf222e", "#8c959f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def figures(o, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    R, data, df = o["results"], o["data"], o["df"]
    chosen = R["chosen_model"]
    pc = o["preds"][chosen]
    oos = (data["open_time"] >= pd.Timestamp(R["meta"]["wf_start"], tz="UTC")) & data["outcome"].notna() & pc["p_target"].notna()
    files = {}

    # 1. reliability of P(target first)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    p, y = pc.loc[oos, "p_target"], (data.loc[oos, "outcome"] == "target").astype(float)
    bins = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
    lim = max(g.max().max(), 0.05) * 1.15
    ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--")
    ax.plot(g["p"], g["y"], "o-", color=ACCENT, ms=4)
    ax.set_xlabel("Predicted chance target hits first"); ax.set_ylabel("Actual frequency")
    ax.set_title("Calibration (walk-forward, deciles)", fontsize=9)
    fig.tight_layout(); files["reliability"] = out / "reliability.png"; fig.savefig(files["reliability"], dpi=200); plt.close(fig)

    # 2. expected-R decile vs realised net R  (does the ranking work?)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    er = o["signals"].loc[oos, "expR"]
    real = (data.loc[oos, "gross_ret"] - COST_RT) / (SL * data.loc[oos, "sigma"])
    dec = pd.qcut(er.rank(method="first"), 10, labels=False) + 1
    gg = pd.DataFrame({"d": dec, "r": real}).groupby("d")["r"].agg(["mean", "sem"])
    ax.bar(gg.index, gg["mean"], yerr=1.96 * gg["sem"], color=[GOOD if v > 0 else BAD for v in gg["mean"]], alpha=0.8,
           error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("Model's expected-result decile (1 = worst)"); ax.set_ylabel("Realised net result (R)")
    ax.set_title("Do higher scores make more money?", fontsize=9)
    fig.tight_layout(); files["deciles"] = out / "deciles.png"; fig.savefig(files["deciles"], dpi=200); plt.close(fig)

    # 3. equity curves, test year
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    a, b = o["periods"]["test"]
    for name, s, col in (("Model signals", o["signals"]["trade"], ACCENT), ("Always take the trade", o["always"], GREY)):
        r, _ = bt.simulate(df, o["labels"], s, a, b)
        ax.plot((1 + r).cumprod(), label=name, color=col, lw=1.4)
    ax.plot((1 + bt.buy_and_hold(df, a, b)).cumprod(), label="Buy and hold", color=INK, lw=1, alpha=0.6)
    ax.axhline(1, color=GREY, lw=0.5); ax.legend(frameon=False, fontsize=8); ax.set_ylabel("Growth of $1")
    ax.set_title("Untouched test year, net of fees and slippage", fontsize=9)
    fig.tight_layout(); files["equity"] = out / "equity_test.png"; fig.savefig(files["equity"], dpi=200); plt.close(fig)

    # 4. signals on price, test year
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    mt = df["open_time"] >= a
    ax.plot(df.loc[mt, "open_time"], df.loc[mt, "close"], color=INK, lw=0.8)
    tr = o["trades_test"]
    if len(tr):
        colors = tr["outcome"].map({"target": GOOD, "stop": BAD, "expiry": GREY})
        px = df.set_index("open_time").loc[tr["decision_time"], "close"].to_numpy()
        ax.scatter(tr["decision_time"], px, c=colors, s=14, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_title("Signals in the test year (green = target, red = stop, grey = timed out)", fontsize=9)
    fig.tight_layout(); files["signals"] = out / "signals_test.png"; fig.savefig(files["signals"], dpi=200); plt.close(fig)
    if "trend_W" in o:
        from .momentum import backtest as mbt
        d, W = o["trend_daily"], o["trend_W"]
        a, b = pd.Timestamp("2018-06-01", tz="UTC"), o["end"]
        fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for name, col, lw in (("buy_hold", INK, 1.0), ("trend_weekly", ACCENT, 1.5)):
            eq = (1 + mbt(d, W[name], a, b)).cumprod()
            lbl = {"buy_hold": "Buy and hold", "trend_weekly": "Trend gauge, checked every Sunday"}[name]
            axs[0].plot(eq, color=col, lw=lw, label=lbl)
            axs[1].plot(eq / eq.cummax() - 1, color=col, lw=lw)
        axs[0].set_yscale("log"); axs[0].legend(frameon=False, fontsize=8); axs[0].set_ylabel("Growth of $1 (log)")
        axs[1].set_ylabel("Drop from peak")
        for x in ("2022-01-01", "2025-06-15"):
            for ax in axs:
                ax.axvline(pd.Timestamp(x, tz="UTC"), color=GREY, lw=0.6, ls="--")
        axs[0].set_title("Trend layer vs buy and hold, 2018-2026, after costs (dashed lines: dev starts, test starts)", fontsize=9)
        fig.tight_layout(); files["trend"] = out / "trend_vs_buyhold.png"; fig.savefig(files["trend"], dpi=200); plt.close(fig)
    return files


def _f(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\u2013"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def results_pdf(o, files, path: Path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    for nm, fn in (("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf")):
        try:
            pdfmetrics.registerFont(TTFont(nm, f"/usr/share/fonts/truetype/dejavu/{fn}"))
        except Exception:
            pass
    body = "DV" if "DV" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold = "DVB" if "DVB" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    S = {"h1": ParagraphStyle("h1", fontName=bold, fontSize=14, leading=18, spaceAfter=4),
         "h2": ParagraphStyle("h2", fontName=bold, fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=3),
         "p": ParagraphStyle("p", fontName=body, fontSize=8.6, leading=11.5, spaceAfter=3)}
    R = o["results"]
    bk, ch = R["backtests"], R["chosen_model"]

    def table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), body, 7.8), ("FONT", (0, 0), (-1, 0), bold, 7.8),
                               ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        return t

    st = []
    tag = ("" if not R["meta"]["synthetic"] else " — SYNTHETIC POSITIVE CONTROL (planted edge)"
           if R["meta"].get("control_edge") else " — SYNTHETIC PLACEBO (no edge exists)")
    st.append(Paragraph(f"Results appendix{tag}", S["h1"]))
    m = R["meta"]
    st.append(Paragraph(
        f"BTC/USDT 4h bars, {m['first_bar'][:10]} to {m['last_bar'][:10]}. Walk-forward from {m['wf_start']} with "
        f"{m['n_folds']} quarterly refits; purge + 6-bar embargo. Test year (untouched during design): from {m['test_start'][:10]}. "
        f"Costs: {100*m['cost_round_trip']:.2f}% per round trip. Chosen model by the pre-registered rule: <b>{ch}</b>. "
        f"Edge validated on dev (kill switch): <b>{'YES' if R['edge_validated'] else 'NO'}</b>.", S["p"]))

    st.append(Paragraph("1. Does the model predict the trade outcome? (out-of-sample)", S["h2"]))
    rows = [["Model", "Period", "AUC target", "AUC stop", "Log-loss", "Brier", "IC exp. R"]]
    for k in ("base", "logreg", "gbm"):
        for per in ("dev", "test"):
            c = R["clf"][k][per]
            rows.append([{"base": "Base rate", "logreg": "Logistic", "gbm": "GBM (depth 3)"}[k], per,
                         _f(c["auc_target"], nd=3), _f(c["auc_stop"], nd=3), _f(c["logloss"], nd=4), _f(c["brier"], nd=4),
                         _f(c["ic_expectedR"], nd=3)])
    st.append(table(rows, [28*mm, 14*mm, 20*mm, 18*mm, 18*mm, 16*mm, 18*mm]))
    c = R["clf"]["base"]["test"]
    st.append(Paragraph(f"Outcome mix in the test year: target first {_f(c['rate_target'], True)}, stop first "
                        f"{_f(c['rate_stop'], True)}, timed out {_f(c['rate_timeout'], True)}. IC = Spearman rank correlation "
                        f"between the model's expected result and the realised net result per bar.", S["p"]))

    st.append(Paragraph("2. Trading results, net of costs (one position at a time, 100% of account, no leverage)", S["h2"]))
    rows = [["Strategy", "Period", "Trades", "Win", "Stop", "Avg net R", "CAGR", "Sharpe", "Max DD", "Exposure"]]
    for key, name in ((f"{ch}|all_signals", "Model: all signals"), (f"{ch}|medium_plus", "Model: Medium+ only"),
                      ("always_take|-", "Always take trade"), ("buy_hold|-", "Buy and hold")):
        for per in ("dev", "test"):
            x = bk[f"{key}|{per}"]
            rows.append([name, per, str(x.get("n_trades", "\u2013")), _f(x.get("win_rate"), True), _f(x.get("stop_rate"), True),
                         _f(x.get("avg_R")), _f(x["cagr"], True), _f(x["sharpe"]), _f(x["max_dd"], True), _f(x["exposure"], True)])
    st.append(table(rows, [32*mm, 12*mm, 14*mm, 15*mm, 15*mm, 19*mm, 17*mm, 15*mm, 18*mm, 18*mm]))
    d = R["deflated_sharpe_dev"]
    st.append(Paragraph(f"Configurations tried: {d['n_trials']}. Deflated Sharpe probability (dev): "
                        f"{_f(d.get('dsr'), nd=3)} (values near 1 mean the Sharpe is unlikely to be luck from trying several variants).", S["p"]))

    rows = [["Costs", "Model trades", "Model CAGR", "Model Sharpe", "Always CAGR", "Always Sharpe"]]
    for cm, v in R["cost_sensitivity_test"].items():
        rows.append([f"{cm}x", str(v["model"]["n_trades"]), _f(v["model"]["cagr"], True), _f(v["model"]["sharpe"]),
                     _f(v["always_take"]["cagr"], True), _f(v["always_take"]["sharpe"])])
    st.append(Spacer(1, 3)); st.append(table(rows, [20*mm, 24*mm, 24*mm, 24*mm, 26*mm, 26*mm]))

    st.append(Spacer(1, 4))
    st.append(Table([[Image(str(files["reliability"]), 60*mm, 53*mm), Image(str(files["deciles"]), 60*mm, 53*mm)]]))
    st.append(Image(str(files["equity"]), 170*mm, 63*mm))
    st.append(Image(str(files["signals"]), 170*mm, 63*mm))

    st.append(Paragraph("3. Honest track record by confidence tier (test year)", S["h2"]))
    rows = [["Tier", "Signal bars", "Target first", "Stop first", "Predicted R", "Realised net R"]]
    for tn in ("Low", "Medium", "High"):
        t = R["track_record_test"][tn]
        rows.append([tn, str(t["n_bars"]), _f(t["target_rate"], True), _f(t["stop_rate"], True),
                     _f(t["avg_predicted_R"]), _f(t["avg_net_R"])])
    st.append(table(rows, [20*mm, 22*mm, 24*mm, 22*mm, 24*mm, 26*mm]))
    st.append(Paragraph(f"Share of test-year bars with no signal: {_f(R['track_record_test']['no_signal_share'], True)}.", S["p"]))

    st.append(Paragraph("4. Robustness", S["h2"]))
    rob = R["robustness"]
    rows = [["Check", "Period", "AUC target", "IC", "Trades", "Sharpe", "Always Sharpe"]]
    for k in ("barriers_x0.75", "barriers_x1.25"):
        for per in ("dev", "test"):
            x = rob[f"{k}|{per}"]
            rows.append([k.replace("barriers_x", "Barriers \u00d7"), per, _f(x["clf"]["auc_target"], nd=3), _f(x["clf"]["ic_expectedR"], nd=3),
                         str(x["model"]["n_trades"]), _f(x["model"]["sharpe"]), _f(x["always_take"]["sharpe"])])
    dt = rob["drop_top_feature"]
    for per in ("dev", "test"):
        rows.append([f"Drop '{dt['feature']}'", per, _f(dt[per]["auc_target"], nd=3), _f(dt[per]["ic_expectedR"], nd=3), "", "", ""])
    st.append(table(rows, [30*mm, 12*mm, 20*mm, 20*mm, 24*mm, 24*mm, 28*mm]))
    rows = [["Year", "AUC target", "IC exp. R", "Signal coverage", "Signal avg net R"]]
    for y, x in rob["by_year"].items():
        rows.append([y, _f(x["auc_target"], nd=3), _f(x["ic_expectedR"], nd=3), _f(x["coverage"], True), _f(x["signal_avg_net_R"])])
    for g, x in rob["by_regime"].items():
        rows.append([f"Regime: {g}", _f(x["auc_target"], nd=3), _f(x["ic_expectedR"], nd=3), "", ""])
    st.append(Spacer(1, 3)); st.append(table(rows, [30*mm, 22*mm, 22*mm, 26*mm, 28*mm]))
    v = R["vol"]
    st.append(Paragraph(f"Volatility model (QLIKE, lower is better): HAR {_f(v['dev']['qlike_har'], nd=3)} vs EWMA "
                        f"{_f(v['dev']['qlike_ewma'], nd=3)} on dev; HAR {_f(v['test']['qlike_har'], nd=3)} vs EWMA "
                        f"{_f(v['test']['qlike_ewma'], nd=3)} on test.", S["p"]))
    top = list(R["coef_target_vs_stop_dev"].items())[:6]
    st.append(Paragraph("Largest logistic weights (target-vs-stop, standardised): " +
                        ", ".join(f"{k} {v:+.3f}" for k, v in top) + ".", S["p"]))
    L = R.get("luck_checks")
    if L:
        st.append(Paragraph("5. Is it luck? (extra checks; none of these change the strategy)", S["h2"]))
        rows = [["Check", "Dev", "Test year"]]
        rd, rt = L["random_entry_dev"], L["random_entry_test"]
        rows.append(["Model vs random entries (same trade count): p-value on Sharpe",
                     f"{_f(rd.get('p_value_sharpe'), nd=3)} ({rd['n_trades']} trades)", f"{_f(rt.get('p_value_sharpe'), nd=3)} ({rt['n_trades']} trades)"])
        rows.append(["Median Sharpe of random entries", _f(rd.get("random_sharpe_median")), _f(rt.get("random_sharpe_median"))])
        rows.append(["Rank-IC (gross) and circular-shift p-value", f"{_f(L['ic_shift_dev']['ic'], nd=3)} (p={_f(L['ic_shift_dev']['p_value'], nd=3)})",
                     f"{_f(L['ic_shift_test']['ic'], nd=3)} (p={_f(L['ic_shift_test']['p_value'], nd=3)})"])
        st.append(table(rows, [90*mm, 42*mm, 42*mm]))
        rows = [["Year", "Trades", "Target", "Stop", "Time-out", "Net return"]]
        for y, v in L["by_year_trades"].items():
            rows.append([y, str(v["trades"]), str(v["target"]), str(v["stop"]), str(v["timeout"]), _f(v["net_return"], True)])
        st.append(Spacer(1, 3)); st.append(table(rows, [20*mm, 20*mm, 20*mm, 20*mm, 22*mm, 24*mm]))
        sb = L.get("same_bar")
        if sb:
            st.append(Paragraph(f"Same-candle ambiguity: {sb['ambiguous']} of {sb['labels']:,} labels ({100*sb['ambiguous_share']:.2f}%) had one 4h candle touch "
                                f"both stop and target (counted as a stop); resolving them with 1h candles would flip {sb.get('would_flip_to_target_with_1h', 'n/a')}.", S["p"]))
    if L and L.get("release_gate"):
        G = L["release_gate"]
        st.append(Paragraph("5b. Release gate for the short-term buy signal (stricter, adopted from an independent review; dev period only)", S["h2"]))
        names = {"at_least_100_trades": f"At least 100 trades ({G['n_trades']})",
                 "lower_bound_above_zero": f"Average net R {G['avg_net_R']:+.3f}, weekly block-bootstrap 95% lower bound {G['bootstrap_lower_95']:+.3f} > 0",
                 "positive_at_double_costs": f"Average net R at double costs {G['avg_net_R_double_costs']:+.3f} > 0",
                 "beats_take_every_trade": f"Sharpe {G['sharpe_model']:.2f} better than taking every trade ({G['sharpe_take_every_trade']:.2f})"}
        rows = [["Check", "Result"]] + [[names[k], "pass" if v else "FAIL"] for k, v in G["checks"].items()]
        st.append(table(rows, [140*mm, 30*mm]))
        st.append(Paragraph(f"Overall: {'PASSED, the card may show buy signals' if G['passed'] else 'FAILED, so the short-term check runs in research mode and never says buy'}.", S["p"]))
        SZ = L.get("sized_short_term", {})
        if SZ:
            rows = [["Short-term strategy with 1%-risk sizing", "Trades", "Total", "Sharpe", "Worst drop", "Avg position"]]
            for per, m in SZ.items():
                if m.get("n_trades"):
                    rows.append([per, str(m["n_trades"]), _f(m["total_return"], True), _f(m["sharpe"]), _f(m["max_dd"], True), _f(m.get("avg_weight_in_trade"), True)])
            st.append(table(rows, [60*mm, 18*mm, 22*mm, 20*mm, 24*mm, 26*mm]))
    TR = R.get("trend")
    if TR:
        st.append(Paragraph("7. The trend layer (time-series momentum, five lookbacks averaged, no tuning)", S["h2"]))
        rows = [["Variant", "Period", "Total", "CAGR", "Sharpe", "Worst drop", "Avg weight"]]
        names = {"buy_hold": "Buy and hold", "trend_daily": "Trend, daily", "trend_weekly": "Trend, Sundays only",
                 "trend_daily_volscaled": "Trend, vol-scaled"}
        for v in ("buy_hold", "trend_weekly", "trend_daily", "trend_daily_volscaled"):
            for per in ("early_2018_2021", "dev_2022_2025", "test_2025_2026", "full_2018_2026"):
                x = TR["variants"][v][per]
                rows.append([names[v], per.split("_", 1)[0], _f(x["total_return"], True), _f(x["cagr"], True), _f(x["sharpe"]),
                             _f(x["max_dd"], True), _f(x["avg_weight"])])
        st.append(table(rows, [34*mm, 16*mm, 22*mm, 20*mm, 18*mm, 22*mm, 20*mm]))
        rows = [["Single lookback only (robustness)", "Early Sharpe", "Dev Sharpe", "Test Sharpe", "Full Sharpe", "Full worst drop"]]
        for v, per in TR["variants"].items():
            if v.endswith("_only"):
                rows.append([v.replace("trend_", "").replace("_only", ""), _f(per["early_2018_2021"]["sharpe"]), _f(per["dev_2022_2025"]["sharpe"]),
                             _f(per["test_2025_2026"]["sharpe"]), _f(per["full_2018_2026"]["sharpe"]), _f(per["full_2018_2026"]["max_dd"], True)])
        st.append(Spacer(1, 3)); st.append(table(rows, [44*mm, 24*mm, 24*mm, 24*mm, 24*mm, 28*mm]))
        bsw = TR["bootstrap_vs_buy_hold"]["trend_weekly"]
        st.append(Paragraph("Sharpe difference vs buy and hold (Sundays-only trend), block bootstrap 95% interval: " + "; ".join(
            f"{k.split('_')[0]} {v['diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in bsw.items()) +
            f". Deflated Sharpe (full period, all trend variants counted): {TR['deflated_sharpe_full']['trend_weekly']['dsr']:.3f}. "
            f"Position changes per year with the Sunday check: {TR['weekly_changes_per_year']:.0f}. Buy and hold in this section is measured "
            f"open to open on daily candles, so it differs slightly from section 2, which uses 4h candles (test year {TR['variants']['buy_hold']['test_2025_2026']['total_return']:+.1%} "
            f"vs {R['backtests']['buy_hold|-|test']['total_return']:+.1%}).", S["p"]))
        if "trend" in files:
            st.append(Image(str(files["trend"]), 170*mm, 107*mm))
    dr = R["meta"].get("data_report")
    if dr:
        st.append(Paragraph("6. Data integrity", S["h2"]))
        st.append(Paragraph(f"Source: {R['meta']['data_source']}. Raw rows {dr.get('rows_raw')}, clean rows {dr.get('rows_clean')}, duplicate-timestamp rows "
                            f"{dr.get('duplicate_timestamp_rows')} (corrupted duplicates kept only if consistent with neighbours), missing 4h bars "
                            f"{dr.get('missing_bars')} (exchange outages, all before March 2020), labels dropped because their 24h window crossed a gap: "
                            f"{dr.get('labels_dropped_for_gaps')}. Every 4h candle was cross-checked against the independent 1h file: all but one corrupted "
                            f"1h row agree within 0.2%.", S["p"]))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm,
                      bottomMargin=14*mm, title="Results appendix", author="Anuj Pavan Sharma").build(st)


def dashboard(o, path: Path):
    df, R = o["df"], o["results"]
    a = o["periods"]["test"][0]
    m = df["open_time"] >= a - pd.Timedelta(days=30)
    ohlc = {"t": (df.loc[m, "open_time"] + pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d %H:%M").tolist(),
            **{k: df.loc[m, k].round(2).tolist() for k in ("open", "high", "low", "close")}}
    payload = {"ohlc": ohlc, "cards": o["cards"], "track": R["track_record_test"], "validated": R["edge_validated"],
               "synthetic": R["meta"]["synthetic"], "control": bool(R["meta"].get("control_edge")), "test_start": str(a)[:10],
               "always": R["backtests"]["always_take|-|test"], "model": R["backtests"][f"{R['chosen_model']}|all_signals|test"],
               "bh": R["backtests"]["buy_hold|-|test"],
               "trend": R.get("trend", {}).get("variants", {}).get("trend_weekly", {}).get("test_2025_2026", R["backtests"]["buy_hold|-|test"])}
    if R.get("trend"):   # compare the trend gauge with buy and hold measured the same way (daily, open to open)
        payload["bh"] = R["trend"]["variants"]["buy_hold"]["test_2025_2026"]
    for c in payload["cards"]:
        c["time"] = c["time"][:16].replace("T", " ")
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text()
    path.write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload, default=str)))
