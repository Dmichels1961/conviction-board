"""Point-in-time backtest of the whole score: does the board's top 25 beat the pool and the S&P 500?

Each month-end T the board is rebuilt using only what was public on T:
  overlap    best ideas: distinct managers holding each stock among a fund's 15 largest positions, from the newest N-PORT filing
             FILED on or before T (not the period it covers). The pool is every stock 2+ funds hold.
  flow       initiations / adds minus trims / exits between each fund's last two filings public by T
  earnings   last 4 reported surprises (size, consistency) and the 2-day price reaction vs SPY, all dated on or before T
  momentum   12-month return excluding the latest month, from prices up to T
Signals with no history are left out and named in the output: consensus growth and revisions (Yahoo keeps no history; our own daily
snapshots start 2026-10-29), guidance, and the SEC fundamentals (to be added from filing dates).

The top 25 by score are held equal-weight for one month (21 trading days, so months never overlap) and compared with the equal-weight
pool (every stock 2+ funds held at T) and with SPY.

In sample / out of sample: weights for the four groups are chosen on the first 60% of months only, from a small grid, by the
t-statistic of monthly excess return. The last 40% of months judge them, along with the model's current weights. Only the
out-of-sample numbers say anything about the future.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as BT

ROOT = Path(__file__).resolve().parent
CURRENT = {"earnings": 20, "overlap": 16, "flow": 5, "momentum": 5}   # earnings: only the 20 of its 60 pts that have history (surprise, consistency, reaction)
BEST_N = 15
EXIT = 35          # matches weekly.Config.exit_rank


def best_ideas_at(hist, T, managers):
    """Distinct managers holding the stock among a fund's 15 largest positions, from filings public by T."""
    by = {}
    for fund, snaps in hist.items():
        vis = [s for s in snaps if s[0] <= T]
        if not vis: continue
        for t in vis[-1][2].sort_values(ascending=False).index[:BEST_N]: by.setdefault(t, set()).add(managers.get(fund, fund))
    return pd.Series({t: len(m) for t, m in by.items()}, dtype=float)


def earnings_at(ev, px, T):
    """Per ticker, as of T: average capped surprise, beat share, standardized surprise, average 2-day reaction vs SPY (last 4 reports)."""
    e = ev[ev.date <= T].sort_values("date").groupby("ticker").tail(4)
    if e.empty: return pd.DataFrame()
    idx = px.index; out = {}
    for t, g in e.groupby("ticker"):
        s = (g.surprise_pct / 100).clip(-0.5, 0.5).dropna()
        reac = []
        if t in px.columns:
            for d in g.date:
                i = idx.searchsorted(d)
                if 1 <= i < len(idx) - 1 and idx[i + 1] <= T:
                    p0, p1, s0, s1 = px[t].iloc[i - 1], px[t].iloc[i + 1], px["SPY"].iloc[i - 1], px["SPY"].iloc[i + 1]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0: reac.append((p1 / p0 - 1) - (s1 / s0 - 1))
        last = g.sort_values("date").iloc[-1]; age = (T - last.date).days
        fresh_w = 1.0 if age <= 60 else max(0.25, 1 - 0.75 * (age - 60) / 120)       # drift is strongest in the first 60 days
        ls = float(np.clip(last.surprise_pct / 100, -0.5, 0.5)) if pd.notna(last.surprise_pct) else np.nan
        out[t] = dict(surprise=s.mean() if len(s) else np.nan, beats=(s > 0).mean() if len(s) else np.nan,
                      sue=s.mean() / max(s.std(), 0.02) if len(s) >= 3 else np.nan, reaction=np.mean(reac) if reac else np.nan,
                      fresh=ls * fresh_w if pd.notna(ls) else np.nan)
    return pd.DataFrame(out).T


MACRO = Path(__file__).resolve().parent / "data/macro/hy_oas.csv"
_OAS = None


def _sector():
    p = Path(__file__).resolve().parent / "data_web/auto.csv"
    try:
        a = pd.read_csv(p); return a.set_index("ticker")["sector"] if "sector" in a.columns else pd.Series(dtype=str)
    except Exception: return pd.Series(dtype=str)


SECTOR = _sector()      # today's sector labels (sectors rarely change; a small, stated approximation)


def active_best_at(hist, T, managers):
    """Active-weight best ideas: managers holding the stock among a fund's 15 largest weights above the peer consensus weight."""
    snaps = {}
    for f, s in hist.items():
        vis = [x for x in s if x[0] <= T]
        if vis: snaps[f] = vis[-1][2]
    if not snaps: return pd.Series(dtype=float)
    W = pd.DataFrame(snaps).fillna(0.0); cons = W.mean(axis=1); by = {}
    for f in W.columns:
        for t in (W[f] - cons)[W[f] > 0].sort_values(ascending=False).index[:BEST_N]: by.setdefault(t, set()).add(managers.get(f, f))
    return pd.Series({t: len(m) for t, m in by.items()}, dtype=float)


RC_LOWVOL, RC_N, RC_EXIT = 0.05, 40, 55


def rc_score(d, w):
    """Risk-controlled board score: each signal ranked within its sector, residual momentum, active-weight best ideas, 5% low volatility."""
    x = d.assign(overlap=d["abi"], momentum=d["resmom"])
    ww = {k: v * (1 - RC_LOWVOL) / sum(w.values()) for k, v in w.items()}; ww["lowvol"] = RC_LOWVOL
    return sum(v * x.groupby("sector")[k].rank(pct=True).fillna(0.5) for k, v in ww.items())


def regime_at(px, i, T):
    """Risk-off when the S&P 500 is below its 200-day average, or high-yield credit spreads (FRED BAMLH0A0HYM2) have widened by more
    than 0.75 points over three months. Both are standard, free stress gauges; only data up to T is used."""
    global _OAS
    spy = px["SPY"].iloc[: i + 1].dropna()
    below = len(spy) >= 200 and spy.iloc[-1] < spy.iloc[-200:].mean()
    wide = False
    if _OAS is None and MACRO.exists():
        o = pd.read_csv(MACRO); o.columns = ["date", "oas"]; o["date"] = pd.to_datetime(o.date)
        _OAS = pd.to_numeric(o.oas, errors="coerce").set_axis(o.date).dropna()
    if _OAS is not None:
        o = _OAS[_OAS.index <= T]
        if len(o) > 70: wide = (o.iloc[-1] - o.iloc[-64]) > 0.75
    return bool(below or wide)


def pr(s):
    return s.rank(pct=True).fillna(0.5)


def panel(hist, px, ev, start=None, managers=None):
    """Signal ranks for every rebalance month: list of (T, DataFrame[ticker -> earnings, overlap, flow, momentum, fwd, fwd_spy])."""
    dates = px.index
    rebal = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).last().values
    out = []
    for T in rebal:
        T = pd.Timestamp(T)
        if start is not None and T < start: continue
        i = dates.get_loc(T)
        if i + 21 >= len(dates) or i < 252: continue
        if pd.isna(px["SPY"].iloc[i - 252]) or pd.isna(px["SPY"].iloc[i + 21]): continue   # data guard: needs a full year of S&P prices
        ov, fl = BT.scores_at(hist, T)
        pool = [t for t in ov[ov >= 2].index if t in px.columns]
        if len(pool) < 60: continue
        fwd = (px.iloc[i + 21][pool] / px.iloc[i][pool] - 1)
        ok = fwd.notna() & px.iloc[i][pool].notna()
        pool = list(fwd[ok].index)
        e = earnings_at(ev, px, T).reindex(pool)
        mom = px.iloc[i - 21][pool] / px.iloc[i - 252][pool] - 1
        d = pd.DataFrame(index=pool)
        d["earnings"] = pd.concat([pr(e.surprise), pr(e.sue), pr(e.beats), pr(e.reaction)], axis=1).mean(axis=1) if len(e.columns) else 0.5
        # fresh-weighted version kept for comparison only (the audit found it weaker on rank correlation)
        d["earnings_fresh"] = pd.concat([pr(e.surprise), pr(e.sue), pr(e.beats), pr(e.reaction), pr(e.fresh), pr(e.fresh)], axis=1).mean(axis=1) if len(e.columns) else 0.5
        r1 = px.iloc[i - 252:i + 1][pool + ["SPY"]].pct_change().iloc[1:]
        beta_ = r1[pool].apply(lambda c: c.cov(r1["SPY"])) / r1["SPY"].var()
        resid = r1[pool] - np.outer(r1["SPY"], beta_)
        d["lowvol"] = pr(-resid.std())                                                     # low idiosyncratic volatility, past year
        rr = resid.iloc[:-21]; d["resmom"] = pr(rr.sum() / rr.std().where(rr.std() > 1e-4))   # residual momentum, months 12 to 2
        d["beta"] = beta_.clip(0, 3)
        d["abi"] = pr(active_best_at(hist, T, managers or {}).reindex(pool).fillna(0))     # active-weight best ideas
        d["sector"] = SECTOR.reindex(pool).fillna("Unknown").values
        d["overlap"] = pr(best_ideas_at(hist, T, managers or {}).reindex(pool).fillna(0)); d["flow"] = pr(fl.reindex(pool).fillna(0)); d["momentum"] = pr(mom)
        d["fwd"] = fwd[pool]; d["fwd_spy"] = float(px["SPY"].iloc[i + 21] / px["SPY"].iloc[i] - 1)
        d["fwd_res"] = d["fwd"] - d["beta"] * d["fwd_spy"]                                 # return left after the market's move x beta
        f = lambda t: float(px[t].iloc[i + 21] / px[t].iloc[i] - 1) if t in px.columns and pd.notna(px[t].iloc[i]) else np.nan
        mq = mom.rank(pct=True); d["fwd_qqq"] = f("QQQ")
        d.attrs["factors"] = dict(mkt=f("SPY"), growth=f("IWF") - f("SPY") if "IWD" not in px.columns else f("IWF") - f("IWD"),
                                  tech=f("QQQ") - f("SPY"), mom=float(fwd.reindex(mq.index)[mq >= 2 / 3].mean() - fwd.reindex(mq.index)[mq <= 1 / 3].mean()))
        d.attrs["risk_off"] = regime_at(px, i, T)
        d["has_earn"] = e.surprise.notna().values if len(e.columns) else False
        out.append((T, d))
    return out


RISK_OFF = {"earnings": 0.4, "overlap": 0.3, "flow": 0.05, "momentum": 0.0, "lowvol": 0.25}


def evaluate(pan, w, top_n=25, exit_rank=None, regime=None, score_fn=None):
    """exit_rank: sell discipline. Names already held stay while they rank exit_rank or better; new names enter only in the top_n."""
    rows, prev = [], set()
    for T, d in pan:
        ww = regime if (regime and d.attrs.get("risk_off")) else w
        s = score_fn(d, ww) if score_fn else sum(d[k] * v for k, v in ww.items())
        order = list(s.sort_values(ascending=False).index)
        if exit_rank and prev:
            keep = [t for t in order[:exit_rank] if t in prev][:top_n]
            pick = keep + [t for t in order if t not in set(keep)][:top_n - len(keep)]
        else:
            pick = order[:top_n]
        top, pool = d.loc[pick, "fwd"].mean(), d["fwd"].mean()
        ic = s.rank().corr(d["fwd"].rank())
        res = d.loc[pick, "fwd_res"].mean() - d["fwd_res"].mean() if "fwd_res" in d.columns else np.nan
        ic_res = s.rank().corr(d["fwd_res"].rank()) if "fwd_res" in d.columns else np.nan
        beta_top = d.loc[pick, "beta"].mean() if "beta" in d.columns else np.nan
        rows.append(dict(date=T, top=top, pool=pool, spy=d["fwd_spy"].iloc[0], ex_pool=top - pool, ex_spy=top - d["fwd_spy"].iloc[0], ex_qqq=(top - d["fwd_qqq"].iloc[0]) if "fwd_qqq" in d.columns else np.nan, ic=ic, ex_adj=res, ic_res=ic_res, beta_top=beta_top,
                         risk_off=d.attrs.get("risk_off", False), **d.attrs.get("factors", {}),
                         turnover=(len(set(pick) - prev) / top_n) if prev else np.nan))
        prev = set(pick)
    return pd.DataFrame(rows)


def stats(r):
    if r.empty: return {}
    n = len(r); t = lambda x: float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
    cum = (1 + r.ex_pool).cumprod(); dd = float((cum / cum.cummax() - 1).min())
    return dict(months=n, first=str(r.date.iloc[0].date()), last=str(r.date.iloc[-1].date()),
                ex_pool_month=round(float(r.ex_pool.mean()), 4), ex_spy_month=round(float(r.ex_spy.mean()), 4),
                ex_pool_ann=round(float((1 + r.ex_pool.mean()) ** 12 - 1), 4), hit_pool=round(float((r.ex_pool > 0).mean()), 2),
                hit_spy=round(float((r.ex_spy > 0).mean()), 2), t_pool=round(t(r.ex_pool), 2), t_spy=round(t(r.ex_spy), 2),
                ic=round(float(r.ic.mean()), 3), max_dd_vs_pool=round(dd, 4), turnover=round(float(r.turnover.mean()), 2) if r.turnover.notna().any() else None,
                ex_adj_month=round(float(r.ex_adj.mean()), 4) if "ex_adj" in r and r.ex_adj.notna().any() else None,
                t_adj=round(t(r.ex_adj.dropna()), 2) if "ex_adj" in r and r.ex_adj.notna().sum() > 2 else None,
                ir_adj=round(float(r.ex_adj.mean() / r.ex_adj.std() * np.sqrt(12)), 2) if "ex_adj" in r and r.ex_adj.notna().sum() > 2 else None,
                t_ic=round(t(r.ic.dropna()), 2), ic_res=round(float(r.ic_res.mean()), 3) if "ic_res" in r else None,
                t_ic_res=round(t(r.ic_res.dropna()), 2) if "ic_res" in r and r.ic_res.notna().sum() > 2 else None,
                beta=round(float(r.beta_top.mean()), 2) if "beta_top" in r and r.beta_top.notna().any() else None,
                ex_qqq_month=round(float(r.ex_qqq.mean()), 4) if "ex_qqq" in r and r.ex_qqq.notna().any() else None,
                t_qqq=round(t(r.ex_qqq.dropna()), 2) if "ex_qqq" in r and r.ex_qqq.notna().sum() > 2 else None,
                hit_qqq=round(float((r.ex_qqq.dropna() > 0).mean()), 2) if "ex_qqq" in r and r.ex_qqq.notna().any() else None,
                ir_spy=round(float(r.ex_spy.mean() / r.ex_spy.std() * np.sqrt(12)), 2) if r.ex_spy.notna().sum() > 2 else None,
                ir_qqq=round(float(r.ex_qqq.mean() / r.ex_qqq.std() * np.sqrt(12)), 2) if "ex_qqq" in r and r.ex_qqq.notna().sum() > 2 else None)


def alpha(r):
    """Monthly return of the top 25 regressed on the market (SPY), growth (IWF minus IWD, or IWF minus SPY), tech (QQQ minus SPY) and
    momentum (top minus bottom third of the pool on 12-1 momentum). The intercept is the return not explained by those tilts."""
    cols = [c for c in ("mkt", "growth", "tech", "mom") if c in r.columns]
    d = r[["top"] + cols].dropna()
    if len(d) < 18: return {}
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols]); y = d["top"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    se = np.sqrt(np.diag((e @ e / (len(d) - X.shape[1])) * np.linalg.inv(X.T @ X)))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    out = dict(alpha_month=round(float(b[0]), 4), alpha_ann=round(float((1 + b[0]) ** 12 - 1), 4), t_alpha=round(float(b[0] / se[0]), 2),
               r2=round(float(r2), 2), months=len(d))
    out.update({f"beta_{c}": round(float(v), 2) for c, v in zip(cols, b[1:])})
    out.update({f"t_{c}": round(float(v), 1) for c, v in zip(cols, b[1:] / se[1:])})
    return out


def grid():
    seen = set()
    for e, o, f, m in itertools.product([0.3, 0.5, 0.7], [0.1, 0.3, 0.5], [0, 0.1, 0.2], [0, 0.1, 0.2]):
        tot = e + o + f + m; k = tuple(round(x / tot, 3) for x in (e, o, f, m))
        if k not in seen: seen.add(k); yield dict(zip(["earnings", "overlap", "flow", "momentum"], k))


def norm(w):
    t = sum(w.values()); return {k: round(v / t, 3) for k, v in w.items()}


def compare(pan, ins, oos, cur):
    """Main board (top 25) and the risk-controlled board (top 40): raw and beta-adjusted return, rank correlation, alpha, risk."""
    fresh = {**cur}; fresh["earnings_fresh"] = fresh.pop("earnings")
    v = {"Main board (25)": (cur, None, 25, EXIT), "Risk-controlled board (40)": (cur, rc_score, RC_N, RC_EXIT),
         "Main board with fresh-weighted earnings": (fresh, None, 25, EXIT)}
    out = {}
    for name, (w, fn, n, ex) in v.items():
        reg = None
        rf = evaluate(pan, w, n, ex, score_fn=fn); ro = evaluate(oos, w, n, ex, score_fn=fn); ri = evaluate(ins, w, n, ex, score_fn=fn)
        off = rf[rf.risk_off] if "risk_off" in rf.columns else rf.iloc[:0]
        out[name] = dict(full=stats(rf), in_sample=stats(ri), out_of_sample=stats(ro), alpha=alpha(rf),
                         risk_off=dict(months=len(off), top=round(float(off.top.mean()), 4) if len(off) else None,
                                       spy=round(float(off.spy.mean()), 4) if len(off) else None, ex_pool=round(float(off.ex_pool.mean()), 4) if len(off) else None))
    return out


def paper_rules(pan, w, n=10, cap=3, cost=0.001):
    """The paper portfolio's rules on history: main-board score, top 10 with at most 3 per sector, equal weight, held two months
    (rebalanced every other month-end, about 60 days), 0.10% cost per trade. Compared with SPY and QQQ held over the same windows."""
    rows, prev = [], set(); nav = spy = qqq = 1.0; series = []
    for j in range(0, len(pan) - 1, 2):
        T, d = pan[j]; T2, d2 = pan[j + 1]
        s = sum(d[k] * v for k, v in w.items()).sort_values(ascending=False); out, per = [], {}
        for t in s.index:
            sec = d.loc[t, "sector"]
            if sec != "Unknown" and per.get(sec, 0) >= cap: continue
            out.append(t); per[sec] = per.get(sec, 0) + 1
            if len(out) == n: break
        r1 = d.loc[out, "fwd"]; r2 = d2["fwd"].reindex(out).fillna(d2["fwd"].mean())
        port = float(((1 + r1) * (1 + r2) - 1).mean())
        turn = len(set(out) - prev) / n if prev else 1.0; port -= 2 * turn * cost; prev = set(out)
        b = lambda c: float((1 + d[c].iloc[0]) * (1 + d2[c].iloc[0]) - 1) if c in d.columns and pd.notna(d[c].iloc[0]) and pd.notna(d2[c].iloc[0]) else np.nan
        rs, rq = b("fwd_spy"), b("fwd_qqq")
        nav *= 1 + port; spy *= 1 + (rs if rs == rs else 0); qqq *= 1 + (rq if rq == rq else 0)
        rows.append(dict(date=str(T.date()), port=port, spy=rs, qqq=rq)); series.append([str(T2.date()), round(nav, 4), round(spy, 4), round(qqq, 4)])
    r = pd.DataFrame(rows)
    if r.empty: return {}
    tt = lambda x: float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan
    yrs = len(r) / 6; cum = pd.Series([v[1] for v in series]); dd = float((cum / cum.cummax() - 1).min())
    es, eq = (r.port - r.spy).dropna(), (r.port - r.qqq).dropna()
    return dict(periods=len(r), first=rows[0]["date"], last=series[-1][0], ann=round(nav ** (1 / yrs) - 1, 4), ann_spy=round(spy ** (1 / yrs) - 1, 4),
                ann_qqq=round(qqq ** (1 / yrs) - 1, 4), total=round(nav - 1, 4), total_spy=round(spy - 1, 4), total_qqq=round(qqq - 1, 4),
                t_spy=round(tt(es), 2), t_qqq=round(tt(eq), 2), hit_spy=round(float((es > 0).mean()), 2), hit_qqq=round(float((eq > 0).mean()), 2),
                max_drawdown=round(dd, 4), series=series,
                note="Two-month holding windows from month-end; history uses today's sector labels and only stocks still listed, so it is optimistic.")


def run(hist, px, ev, managers=None):
    pan = panel(hist, px, ev, managers=managers)
    if len(pan) < 10: return dict(error=f"only {len(pan)} months of history; need 10+"), pan
    cut = int(len(pan) * 0.6); ins, oos = pan[:cut], pan[cut:]
    tried = []
    for w in grid():
        r = evaluate(ins, w); tried.append((stats(r).get("t_pool", -9) if not r.empty else -9, w))
    tried.sort(key=lambda x: -x[0] if pd.notna(x[0]) else 9)
    best = tried[0][1]; cur = norm(CURRENT)
    res = dict(
        method=__doc__.split("\n\n")[1].replace("\n", " "), months_total=len(pan), split=str(oos[0][0].date()) if oos else None,
        missing="Not in this test (no point-in-time history): consensus growth and estimate revisions, guidance, SEC fundamentals, short interest, insider buying.",
        current_weights=cur, tuned_weights=best, grid_size=len(tried),
        full=dict(current=stats(evaluate(pan, cur, exit_rank=EXIT)), no_buffer=stats(evaluate(pan, cur))),
        in_sample=dict(current=stats(evaluate(ins, cur, exit_rank=EXIT)), tuned=stats(evaluate(ins, best, exit_rank=EXIT))),
        out_of_sample=dict(current=stats(evaluate(oos, cur, exit_rank=EXIT)), tuned=stats(evaluate(oos, best, exit_rank=EXIT)), no_buffer=stats(evaluate(oos, cur))),
        versions=compare(pan, ins, oos, cur),
        risk_off_months=int(sum(bool(d.attrs.get("risk_off")) for _, d in pan)),
        single=dict((k, dict(in_sample=stats(evaluate(ins, {k: 1})), out_of_sample=stats(evaluate(oos, {k: 1})))) for k in ["earnings", "overlap", "flow", "momentum"]),
        monthly=[[str(r.date.date()), round(r.top, 4), round(r.pool, 4), round(r.spy, 4)] for r in evaluate(pan, cur, exit_rank=EXIT).itertuples()])
    return res, pan


def main():
    hist = BT.load_history(ROOT / "data/holdings_history"); px = BT.load_prices(ROOT / "data/prices")
    ev = pd.read_csv(ROOT / "data/earnings_events.csv"); ev["date"] = pd.to_datetime(ev.date, errors="coerce").dt.tz_localize(None).dt.normalize()
    ev = ev.dropna(subset=["date"])
    reg = pd.read_csv(ROOT / "fund_registry.csv"); managers = dict(zip(reg.etf, reg.manager))
    res, pan = run(hist, px, ev, managers)
    try:
        pr_ = paper_rules(pan, norm(CURRENT)); (ROOT / "data/paper").mkdir(parents=True, exist_ok=True)
        (ROOT / "data/paper/backtest.json").write_text(json.dumps(pr_, default=str)); res["paper_rules"] = {k: v for k, v in pr_.items() if k != "series"}
    except Exception as e: res["paper_rules_error"] = str(e)
    for p in (ROOT / "data/pit_backtest.json", ROOT / "docs/data/pit.json"):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, default=str))
    return res


def selftest():
    hist, px = BT.synthetic(True, months=40)
    px["SPY"] = px.mean(axis=1)
    ev = pd.DataFrame({"ticker": ["S000"], "date": [px.index[300]], "surprise_pct": [10.0]})
    e = earnings_at(ev, px, px.index[299]); assert e.empty                 # a report after T is invisible at T
    e = earnings_at(ev, px, px.index[302]); assert abs(e.loc["S000", "surprise"] - 0.1) < 1e-9
    pan = panel(hist, px, ev); assert len(pan) >= 10
    r = stats(evaluate(pan, {"overlap": 1})); assert r["ex_pool_month"] > 0, r      # the synthetic world rewards overlap
    print("pit_backtest selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest(); sys.exit()
    r = main(); print(json.dumps({k: r[k] for k in r if k not in ("monthly", "method")}, indent=1, default=str))
