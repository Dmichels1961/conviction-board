"""Test the signals that carry most of the score on real history, not just overlap.

  python factor_backtest.py            uses data/prices/*.csv (daily closes, incl. SPY) and data/earnings_events.csv (Yahoo, via refresh --history)
  python factor_backtest.py --selftest synthetic worlds with and without an effect; the test must tell them apart

Tests (all out of the stocks funds hold, all vs SPY):
  surprise   earnings-surprise % at each report; entry = close of the day AFTER the report (the reaction day is excluded, so this is drift only)
  momentum   12-1 month return at each month-end
  high52     price / 52-week high at each month-end
Each is sorted into quintiles per period; we report the top-minus-bottom quintile spread over 21 and 63 trading days, a t-statistic across
periods (calendar quarters for earnings, months for the rest; 63-day windows sampled so they do not overlap) and the rank correlation.
Robustness (added after seeing heavy-tailed returns in the first run): returns are winsorised at the 2nd/98th percentile, surprises
are capped at +/-50% and events with an estimate under $0.10 are dropped; the median spread is reported next to the mean.
Biases: only stocks that still trade (survivorship), today's fund holdings define the universe, and Yahoo only keeps ~3 years of report dates.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HORIZONS = (21, 63)


def load_prices(d):
    cols = {}
    for p in Path(d).glob("*.csv"):
        x = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
        cols[p.stem.upper()] = x[~x.index.duplicated()]
    return pd.DataFrame(cols).sort_index()


def fwd(px, bench, i, h, tick):
    """Excess return of tick from day i to i+h vs benchmark."""
    if i + h >= len(px.index): return np.nan
    a, b = px[tick].iloc[i], px[tick].iloc[i + h]
    if not (a > 0 and b > 0): return np.nan
    return b / a - 1 - (bench.iloc[i + h] / bench.iloc[i] - 1)


def quintile_stats(df, score, ret, period):
    df = df.copy(); lo, hi = df[ret].quantile([0.02, 0.98]); df[ret] = df[ret].clip(lo, hi)   # a handful of 5x rebounders must not decide the answer
    """df has score, ret, period. -> spread (Q5-Q1 mean excess), t across periods, IC, n periods."""
    rows = []
    for per, g in df.dropna(subset=[score, ret]).groupby(period):
        if len(g) < 25: continue
        q = pd.qcut(g[score].rank(method="first"), 5, labels=False)
        rows.append(dict(period=per, spread=g.loc[q == 4, ret].mean() - g.loc[q == 0, ret].mean(), mspread=g.loc[q == 4, ret].median() - g.loc[q == 0, ret].median(), top=g.loc[q == 4, ret].mean(), ic=g[score].rank().corr(g[ret].rank()), n=len(g)))
    if len(rows) < 3: return None
    r = pd.DataFrame(rows).sort_values("period")
    return r


def summarize(r, step):
    r = r.iloc[::step]
    t = r.spread.mean() / (r.spread.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.spread.std(ddof=1) > 0 else np.nan
    return dict(spread_pct=round(100 * r.spread.mean(), 2), median_spread_pct=round(100 * r.mspread.mean(), 2), top_pct=round(100 * r.top.mean(), 2), t=round(float(t), 2), ic=round(r.ic.mean(), 3), periods=int(len(r)), hit=round(float((r.spread > 0).mean()), 2))


def run(px, events, bench="SPY"):
    out = []
    if bench not in px.columns: raise SystemExit("SPY prices missing")
    b = px[bench].ffill(); dates = px.index
    # ---- earnings surprise
    if events is not None and len(events):
        ev = events.dropna(subset=["surprise_pct"]).copy()
        if "eps_est" in ev.columns: ev = ev[ev.eps_est.abs() >= 0.10]      # surprise % is meaningless when the estimate is near zero
        ev["surprise_pct"] = ev.surprise_pct.clip(-50, 50); ev["date"] = pd.to_datetime(ev["date"]).dt.tz_localize(None).dt.normalize()
        for h in HORIZONS:
            rows = []
            for e in ev.itertuples():
                if e.ticker not in px.columns: continue
                i = dates.searchsorted(e.date, side="right")        # first trading day after the report date
                if i >= len(dates): continue
                rows.append(dict(ticker=e.ticker, score=float(e.surprise_pct), ret=fwd(px, b, i, h, e.ticker), quarter=f"{e.date.year}Q{(e.date.month - 1) // 3 + 1}"))
            df = pd.DataFrame(rows)
            if len(df):
                r = quintile_stats(df, "score", "ret", "quarter")
                if r is not None: out.append(dict(factor="Earnings surprise (drift after the report)", h=h, **summarize(r, max(1, round(h / 63)))))
    # ---- month-end price factors
    me = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).last().values
    for name, fn, label in [("momentum", lambda i, t: px[t].iloc[i - 21] / px[t].iloc[i - 252] - 1, "Momentum (12-1 months)"),
                            ("high52", lambda i, t: px[t].iloc[i] / px[t].iloc[i - 252:i + 1].max(), "Nearness to 52-week high")]:
        for h in HORIZONS:
            rows = []
            for T in me:
                i = dates.get_loc(T)
                if i < 260 or i + h >= len(dates): continue
                for t in px.columns:
                    if t == bench or not (px[t].iloc[i - 252] > 0 and px[t].iloc[i] > 0): continue
                    rows.append(dict(ticker=t, score=fn(i, t), ret=fwd(px, b, i, h, t), month=T))
            df = pd.DataFrame(rows)
            if len(df):
                r = quintile_stats(df, "score", "ret", "month")
                if r is not None: out.append(dict(factor=label, h=h, **summarize(r, max(1, round(h / 21)))))
    return out


def verdict(row):
    if row["t"] > 2 and row["spread_pct"] > 0: return "Worked"
    if row["t"] < -2: return "Reversed"
    if abs(row["t"]) < 1: return "No effect seen"
    return "Weak / unclear"


def synthetic(effect, seed=3, n=160, years=4):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=252 * years)
    names = [f"T{i:03d}" for i in range(n)]
    q = rng.normal(0, 1, n)                                    # hidden quality: drives both surprises and later drift
    drift = (0.0004 * q if effect else 0 * q)
    ret = rng.normal(0.0004, 0.015, (len(days), n)) + drift
    px = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=days, columns=names)
    px["SPY"] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.008, len(days)))
    rows = []
    for k in range(years * 4):
        d = days[min(k * 63 + 30, len(days) - 3)]
        for j, t in enumerate(names): rows.append(dict(ticker=t, date=d, surprise_pct=float(q[j] * 5 + rng.normal(0, 4))))
    return px, pd.DataFrame(rows)


def selftest():
    for eff in (True, False):
        px, ev = synthetic(eff)
        res = {(r["factor"][:8], r["h"]): r for r in run(px, ev)}
        s = res[("Earnings", 63)]
        assert (s["t"] > 2) is eff, (eff, res)
    print("factor_backtest selftest ok")


def main():
    px = load_prices(ROOT / "data/prices")
    ep = ROOT / "data/earnings_events.csv"
    events = pd.read_csv(ep) if ep.exists() else None
    res = run(px, events)
    for r in res: r["verdict"] = verdict(r)
    out = ROOT / "docs/data/factors.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=res, tickers=int(px.shape[1]), events=0 if events is None else int(events.surprise_pct.notna().sum()),
                                   first=str(px.index.min().date()), last=str(px.index.max().date())), separators=(",", ":")))
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
