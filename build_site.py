"""Builds the public dashboard data (docs/data/*.json) from the daily rankings, the backtest and the signal list.
Runs at the end of every refresh; docs/index.html (GitHub Pages) reads these files, so the page updates itself."""
import json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import weekly as W

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs/data"
W_BY_KEY = {s[0]: s[3] for s in W.SIGNALS}


def clean(v):
    if isinstance(v, (np.floating, float)): return None if pd.isna(v) or np.isinf(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    return None if (not isinstance(v, str) and pd.isna(v)) else v


def daily_files():
    fs = sorted((ROOT / "data/daily").glob("20??-??-??.csv"))
    return [(pd.Timestamp(f.stem), f) for f in fs]


def evidence():
    w = lambda *k: sum(W_BY_KEY.get(x, 0) for x in k)
    E = [
     dict(metric="Earnings surprise and drift", what="Stocks that beat estimates keep drifting up for weeks to months (post-earnings-announcement drift).", paper="Ball & Brown 1968; Bernard & Thomas 1989", strength="Strong", note="Much smaller in large caps than in the 1980s. GAAP-only surprises are capped.", source="Alpha Vantage earnings history (free)", model=f"In model: surprise + consistency, {w('earn_surprise','earn_consistency')} pts"),
     dict(metric="Analyst estimate revisions", what="Upward revisions to consensus EPS predict outperformance over the next 3 to 6 months.", paper="Chan, Jegadeesh & Lakonishok 1996; Gleason & Lee 2003", strength="Strong", note="Needs a daily history, which we are now building ourselves.", source="Yahoo consensus (own daily snapshots)", model=f"In model: revisions, {w('earn_revisions')} pts"),
     dict(metric="Earnings-day price reaction", what="The market's reaction in the days around a report carries information about the news and persists.", paper="Chan, Jegadeesh & Lakonishok 1996", strength="Strong", note="Noisy for a single quarter.", source="Prices around report dates", model=f"In model: reaction, {w('earn_reaction')} pts"),
     dict(metric="Earnings growth and acceleration", what="Companies whose growth is speeding up tend to be rewarded; estimates lag the trend.", paper="Related to earnings momentum literature (Jegadeesh & Livnat 2006)", strength="Moderate", note="Growth is widely known, so much is priced.", source="Yahoo / company reports", model=f"In model: growth, acceleration, sales, guidance, {w('earn_growth','earn_accel','earn_sales','earn_guidance')} pts"),
     dict(metric="Price momentum (12 months, skipping the latest)", what="Winners over the past year keep outperforming losers over the next few months.", paper="Jegadeesh & Titman 1993", strength="Strong", note="Suffers sharp crashes when markets rebound.", source="Daily prices (Yahoo)", model=f"In model: momentum, {w('momentum')} pts"),
     dict(metric="Proximity to the 52-week high", what="Stocks near their 52-week high keep outperforming; investors under-react to good news.", paper="George & Hwang 2004", strength="Moderate", note="Overlaps with momentum.", source="Daily prices (Yahoo)", model="Collected, not yet scored"),
     dict(metric="Manager conviction and overlap", what="Fund managers' largest, most active positions beat the rest of their portfolios; changes in holdings carry information.", paper="Cohen, Polk & Silli 2010; Chen, Jegadeesh & Wermers 2000", strength="Moderate", note="Holdings are public only with a lag; the backtest here is inconclusive so far.", source="SEC N-PORT and issuer files (free)", model=f"In model: breadth, active weight, flow, {w('breadth','active_weight','flow')} pts"),
     dict(metric="Insider open-market buying", what="Clusters of insiders buying with their own money precede outperformance; sales are mostly uninformative.", paper="Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012", strength="Moderate", note="Rare for large caps. Waiting for first data.", source="SEC Form 4 (free)", model=f"In model: insider buying, {w('insider')} pts"),
     dict(metric="Profitability and quality", what="Highly profitable, cash-generative firms outperform, partly because the market undervalues them.", paper="Novy-Marx 2013; Asness, Frazzini & Pedersen 2019", strength="Moderate", note="Slow-moving; works better as a filter than a timing signal.", source="Company financials (Yahoo)", model=f"In model: quality, {w('quality')} pts"),
     dict(metric="Net share issuance and buybacks", what="Firms issuing shares underperform; firms retiring shares outperform.", paper="Daniel & Titman 2006; Pontiff & Woodgate 2008", strength="Moderate", note="Dilution is why Intel is flagged.", source="Company financials", model="Partly in quality"),
     dict(metric="Short interest", what="Heavily shorted stocks underperform on average; shorts are often right.", paper="Dechow et al. 2001; Rapach et al. 2016", strength="Moderate", note="Squeezes make single names unpredictable.", source="Yahoo / exchange short-interest files", model=f"In model: crowding, {w('crowding')} pts"),
     dict(metric="Low beta and low volatility", what="Low-risk stocks earn more than their risk implies.", paper="Frazzini & Pedersen 2014", strength="Moderate", note="Conflicts with an AI-heavy, high-beta board, so the weight is small.", source="Daily prices", model=f"In model: beta, {w('beta')} pt"),
     dict(metric="Industry and theme momentum", what="Industries that did well keep doing well for months.", paper="Moskowitz & Grinblatt 1999", strength="Moderate", note="Also what makes the board one correlated bet.", source="Prices and our theme list", model=f"In model: theme, {w('theme')} pt"),
     dict(metric="Value (earnings yield, book to market)", what="Cheap stocks outperform over long horizons.", paper="Fama & French 1992", strength="Weak lately", note="Poor for a decade; kept at a low weight on purpose.", source="Yahoo", model=f"In model: valuation, {w('valuation')} pt"),
     dict(metric="Analyst price-target changes", what="Target increases predict small positive drift.", paper="Brav & Lehavy 2003", strength="Weak", note="Targets follow prices.", source="Yahoo", model=f"In model: analyst drift, {w('analyst_drift')} pt"),
     dict(metric="Asset growth and accruals", what="Fast balance-sheet growth and low-quality earnings precede underperformance.", paper="Cooper, Gulen & Schill 2008; Sloan 1996", strength="Weak lately", note="Faded after publication. Not used.", source="SEC financial statements (free)", model="Not used"),
    ]
    return E


def backtest():
    p = ROOT / "data/backtest_summary.csv"
    if not p.exists(): return dict(rows=[], reading="No backtest yet.")
    d = pd.read_csv(p); d.columns = ["signal", "h"] + list(d.columns[2:])
    d["signal"] = d["signal"].ffill()
    rows = [{k: clean(v) for k, v in r.items()} for r in d.to_dict("records")]
    nfil = sum(1 for _ in (ROOT / "data/holdings_history").glob("*/20*.csv")) if (ROOT / "data/holdings_history").exists() else 0
    best = d.sort_values("t_nonoverlap", ascending=False).iloc[0]
    sig = bool((d.t_nonoverlap > 2).any())
    reading = ("At least one test clears a t-statistic of 2, which is the usual bar." if sig else
               f"No test clears a t-statistic of 2, the usual bar for a real effect (best: {best['signal']} at {int(best['h'])} days, t = {best['t_nonoverlap']:.1f}). The picks beat the pool on average, but too inconsistently to rule out luck.")
    return dict(rows=rows, filings=nfil, reading=reading, updated=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = daily_files()
    if not fs: raise SystemExit("no daily rankings yet")
    asof, f = fs[-1]; cur = pd.read_csv(f).sort_values("rank")
    def ranks(i):
        return None if i is None else dict(zip(*[pd.read_csv(fs[i][1])[c] for c in ("ticker", "rank")]))
    prev_i = len(fs) - 2 if len(fs) > 1 else None
    wk_i = next((i for i in range(len(fs) - 2, -1, -1) if (asof - fs[i][0]).days >= 6), None)
    prev, wk = ranks(prev_i), ranks(wk_i)
    hist = {}
    for d, ff in fs[-30:]:
        r = pd.read_csv(ff, usecols=["ticker", "rank"])
        for t, k in zip(r.ticker, r["rank"]): hist.setdefault(t, []).append([d.strftime("%Y-%m-%d"), int(k)])
    keep = ["rank", "ticker", "name", "score", "bucket", "data_coverage", "n_managers", "n_etfs", "fwd_growth", "avg_surprise", "rev_30d", "days_to_earnings", "mom_12_1", "beta", "fwd_pe",
            "insider_buyers", "gaap_only", "bucket_capped", "thesis", "drivers", "rank_p10", "rank_p90", "p_top10", "score_raw"]
    keep = [c for c in keep if c in cur.columns]
    rows = []
    for _, r in cur.iterrows():
        o = {c: clean(r[c]) for c in keep}
        o["prev_rank"] = clean(prev.get(r.ticker)) if prev else None
        o["wk_rank"] = clean(wk.get(r.ticker)) if wk else None
        o["hist"] = hist.get(r.ticker, [])
        fl = []
        if r.get("gaap_only") is True or r.get("gaap_only") == True: fl.append("GAAP-only surprise capped")
        if bool(r.get("bucket_capped", False)): fl.append("Pushed down by theme cap")
        if pd.notna(r.data_coverage) and r.data_coverage < 70: fl.append("Thin data")
        if pd.notna(r.days_to_earnings) and 0 <= r.days_to_earnings <= 14: fl.append(f"Reports in {int(r.days_to_earnings)}d")
        if pd.notna(r.get("share_chg", np.nan)) and r.get("share_chg", 0) > 0.05: fl.append("Share dilution")
        o["flags"] = fl
        g = lambda ks: float(sum(r.get(f"pts_{k}", 0) or 0 for k in ks))
        o["pts"] = dict(earnings=g(["earn_growth", "earn_surprise", "earn_consistency", "earn_revisions", "earn_accel", "earn_sales", "earn_reaction", "earn_guidance"]),
                        overlap=g(["breadth", "active_weight"]), flow=g(["flow"]),
                        other=g(["momentum", "quality", "valuation", "theme", "crowding", "beta", "analyst_drift", "insider"]))
        rows.append(o)
    top = cur[cur["rank"] <= 25]; rest = cur[cur["rank"] > 25]
    sig = [c[4:] for c in cur.columns if c.startswith("sig_")]
    exposure = [dict(signal=k, top25=clean(top[f"sig_{k}"].mean()), rest=clean(rest[f"sig_{k}"].mean()), weight=W_BY_KEY.get(k, 0)) for k in sig]
    moves = None
    if prev:
        t25 = set(top.ticker); p25 = {t for t, k in prev.items() if k <= 25}
        ch = [(t, prev[t] - k) for t, k in zip(cur.ticker, cur["rank"]) if t in prev]
        moves = dict(entered=sorted(t25 - p25), left=sorted(p25 - t25), risers=sorted(ch, key=lambda x: -x[1])[:5], fallers=sorted(ch, key=lambda x: x[1])[:5], compared_to=fs[prev_i][0].strftime("%Y-%m-%d"))
    ai = {"AI hardware / semis", "AI platforms / software", "AI infrastructure (network/power)"}
    meta = dict(asof=asof.strftime("%Y-%m-%d"), generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), n_scored=len(cur),
                avg_cov=clean(top.data_coverage.mean()), ai_share=clean(top.bucket.isin(ai).mean()), runs=len(fs),
                funds=int(len(pd.read_csv(ROOT / "fund_registry.csv"))), weights=W_BY_KEY)
    for name, obj in dict(board=dict(meta=meta, rows=rows, moves=moves, exposure=exposure), signals=dict(items=evidence(), decay="McLean & Pontiff (2016) studied 97 published return predictors: returns were about 26% lower out of sample and about 58% lower after publication. Treat every effect as smaller than the papers suggest."), backtest=backtest()).items():
        (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), default=clean))
    print("site data:", meta["asof"], len(rows), "rows")


if __name__ == "__main__":
    main()
