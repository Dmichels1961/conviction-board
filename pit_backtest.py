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
        out[t] = dict(surprise=s.mean() if len(s) else np.nan, beats=(s > 0).mean() if len(s) else np.nan,
                      sue=s.mean() / max(s.std(), 0.02) if len(s) >= 3 else np.nan, reaction=np.mean(reac) if reac else np.nan)
    return pd.DataFrame(out).T


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
        d["overlap"] = pr(best_ideas_at(hist, T, managers or {}).reindex(pool).fillna(0)); d["flow"] = pr(fl.reindex(pool).fillna(0)); d["momentum"] = pr(mom)
        d["fwd"] = fwd[pool]; d["fwd_spy"] = float(px["SPY"].iloc[i + 21] / px["SPY"].iloc[i] - 1)
        d["has_earn"] = e.surprise.notna().values if len(e.columns) else False
        out.append((T, d))
    return out


def evaluate(pan, w, top_n=25, exit_rank=None):
    """exit_rank: sell discipline. Names already held stay while they rank exit_rank or better; new names enter only in the top_n."""
    rows, prev = [], set()
    for T, d in pan:
        s = sum(d[k] * v for k, v in w.items())
        order = list(s.sort_values(ascending=False).index)
        if exit_rank and prev:
            keep = [t for t in order[:exit_rank] if t in prev][:top_n]
            pick = keep + [t for t in order if t not in set(keep)][:top_n - len(keep)]
        else:
            pick = order[:top_n]
        top, pool = d.loc[pick, "fwd"].mean(), d["fwd"].mean()
        ic = s.rank().corr(d["fwd"].rank())
        rows.append(dict(date=T, top=top, pool=pool, spy=d["fwd_spy"].iloc[0], ex_pool=top - pool, ex_spy=top - d["fwd_spy"].iloc[0], ic=ic,
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
                ic=round(float(r.ic.mean()), 3), max_dd_vs_pool=round(dd, 4), turnover=round(float(r.turnover.mean()), 2) if r.turnover.notna().any() else None)


def grid():
    seen = set()
    for e, o, f, m in itertools.product([0.3, 0.5, 0.7], [0.1, 0.3, 0.5], [0, 0.1, 0.2], [0, 0.1, 0.2]):
        tot = e + o + f + m; k = tuple(round(x / tot, 3) for x in (e, o, f, m))
        if k not in seen: seen.add(k); yield dict(zip(["earnings", "overlap", "flow", "momentum"], k))


def norm(w):
    t = sum(w.values()); return {k: round(v / t, 3) for k, v in w.items()}


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
        single=dict((k, dict(in_sample=stats(evaluate(ins, {k: 1})), out_of_sample=stats(evaluate(oos, {k: 1})))) for k in ["earnings", "overlap", "flow", "momentum"]),
        monthly=[[str(r.date.date()), round(r.top, 4), round(r.pool, 4), round(r.spy, 4)] for r in evaluate(pan, cur, exit_rank=EXIT).itertuples()])
    return res, pan


def main():
    hist = BT.load_history(ROOT / "data/holdings_history"); px = BT.load_prices(ROOT / "data/prices")
    ev = pd.read_csv(ROOT / "data/earnings_events.csv"); ev["date"] = pd.to_datetime(ev.date, errors="coerce").dt.tz_localize(None).dt.normalize()
    ev = ev.dropna(subset=["date"])
    reg = pd.read_csv(ROOT / "fund_registry.csv"); managers = dict(zip(reg.etf, reg.manager))
    res, _ = run(hist, px, ev, managers)
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
