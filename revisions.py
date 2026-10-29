"""Estimate revisions, built from our own daily snapshots of consensus EPS (free; no paid feed gives this history).

  snapshot(tickers, today)  once a day: save consensus EPS for the current and next fiscal year (yfinance/Yahoo), plus the number of
                            analysts revising up/down in the last 30 days, into data/consensus/snapshots.csv
  rev_30d(snaps, today)     % change in consensus EPS over ~30 days, from OUR history once it covers 25+ days; before that, Yahoo's own
                            "30 days ago" figure that the snapshot also stores, so the signal works from day one and becomes independent later.
  Revision breadth (up minus down over total) is stored too, but is not scored until the backtest says it adds something.
"""
import time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data/consensus/snapshots.csv"
COLS = ["date", "ticker", "period", "eps", "eps_30d_yahoo", "n_analysts", "up30", "down30", "eps_growth", "rev_growth"]


def fetch_one(t):
    import yfinance as yf
    tk = yf.Ticker(t); tr = tk.eps_trend; rv = None
    try: rv = tk.eps_revisions
    except Exception: pass
    rows = []
    for per in ("0y", "+1y"):
        if per not in tr.index: continue
        up = rv.loc[per].get("upLast30days", np.nan) if rv is not None and per in rv.index else np.nan     # missing for some foreign listings
        dn = rv.loc[per].get("downLast30days", np.nan) if rv is not None and per in rv.index else np.nan
        na = eg = rg = np.nan
        try: ee = tk.earnings_estimate; na = ee.loc[per, "numberOfAnalysts"]; eg = ee.loc[per].get("growth", np.nan)
        except Exception: pass
        try: rg = tk.revenue_estimate.loc[per].get("growth", np.nan)       # same Yahoo response, no extra request
        except Exception: pass
        rows.append(dict(ticker=t, period=per, eps=tr.loc[per, "current"], eps_30d_yahoo=tr.loc[per, "30daysAgo"], n_analysts=na, up30=up, down30=dn,
                         eps_growth=eg, rev_growth=rg))
    return rows


def snapshot(tickers, today=None, log=print, fetch=fetch_one, sleep=0.4):
    today = today or date.today(); rows = []
    for t in tickers:
        try: rows += [dict(date=str(today), **r) for r in fetch(t)]
        except Exception as e: log(f"consensus {t}: {e}")
        time.sleep(sleep)
    new = pd.DataFrame(rows, columns=COLS)
    old = pd.read_csv(FILE) if FILE.exists() else pd.DataFrame(columns=COLS)
    allp = pd.concat([old[old.date != str(today)], new], ignore_index=True)     # re-running the same day replaces, never duplicates
    FILE.parent.mkdir(parents=True, exist_ok=True); allp.to_csv(FILE, index=False)
    log(f"consensus: {len(new)} rows for {len(tickers)} names; history spans {allp.date.nunique()} days")
    return allp


def rev_30d(snaps, today):
    """-> DataFrame(index ticker; rev_30d, rev_breadth, rev_source). Average of the current-year and next-year EPS changes."""
    s = snaps.copy(); s["date"] = pd.to_datetime(s.date); today = pd.Timestamp(today); out = {}
    for t, g in s.groupby("ticker"):
        chg, src = [], "own"
        for per, gp in g.groupby("period"):
            gp = gp.sort_values("date"); cur = gp[gp.date <= today].iloc[-1]
            old = gp[gp.date <= today - pd.Timedelta(days=25)]
            if len(old) and pd.notna(old.iloc[-1].eps) and old.iloc[-1].eps != 0 and pd.notna(cur.eps):
                o = old.iloc[-1]; days = (cur.date - o.date).days
                if days <= 45: chg.append((cur.eps - o.eps) / abs(o.eps)); continue
            if pd.notna(cur.eps_30d_yahoo) and cur.eps_30d_yahoo != 0 and pd.notna(cur.eps):
                chg.append((cur.eps - cur.eps_30d_yahoo) / abs(cur.eps_30d_yahoo)); src = "yahoo"
        last = g[g.date == g.date.max()]
        up, dn = last.up30.sum(min_count=1), last.down30.sum(min_count=1)
        br = (up - dn) / (up + dn) if pd.notna(up) and pd.notna(dn) and up + dn > 0 else np.nan
        if chg: out[t] = dict(rev_30d=float(np.mean(chg)), rev_breadth=br, rev_source=src)
    return pd.DataFrame(out).T


def growth_latest(snaps):
    """Latest consensus growth per ticker: this year's and next year's EPS growth, next year's revenue growth."""
    s = snaps[snaps.date == snaps.date.max()]
    if "eps_growth" not in s.columns: return pd.DataFrame()
    p = s.pivot_table(index="ticker", columns="period", values=["eps_growth", "rev_growth"], aggfunc="last")
    g = lambda a, b: p[(a, b)] if (a, b) in p.columns else pd.Series(np.nan, index=p.index)
    return pd.DataFrame({"fwd_growth": g("eps_growth", "+1y"), "cur_growth": g("eps_growth", "0y"), "rev_growth": g("rev_growth", "+1y")})


def selftest():
    import tempfile
    global FILE
    FILE = Path(tempfile.mkdtemp()) / "s.csv"
    def fake(day_eps):
        return lambda t: [dict(ticker=t, period="0y", eps=day_eps, eps_30d_yahoo=1.0, n_analysts=20, up30=6, down30=2),
                          dict(ticker=t, period="+1y", eps=day_eps * 1.2, eps_30d_yahoo=1.2, n_analysts=18, up30=5, down30=1)]
    q = lambda *a: None
    snapshot(["X"], date(2026, 9, 1), q, fake(1.00), 0)
    snapshot(["X"], date(2026, 9, 1), q, fake(1.00), 0)                     # same day twice: no duplicates
    a = snapshot(["X"], date(2026, 9, 2), q, fake(1.01), 0)
    assert len(a) == 4
    r = rev_30d(a, date(2026, 9, 2)); assert r.loc["X", "rev_source"] == "yahoo" and abs(r.loc["X", "rev_30d"] - 0.01) < 1e-9   # too little own history: Yahoo's figure
    a = snapshot(["X"], date(2026, 10, 1), q, fake(1.10), 0)
    r = rev_30d(a, date(2026, 10, 1)); assert r.loc["X", "rev_source"] == "own" and abs(r.loc["X", "rev_30d"] - (1.10 - 1.01) / 1.01) < 1e-9, r   # 30 days of our own history
    assert abs(r.loc["X", "rev_breadth"] - (11 - 3) / 14) < 1e-9
    print("revisions selftest ok")


if __name__ == "__main__":
    selftest()
