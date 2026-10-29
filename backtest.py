"""Does manager overlap (and manager flow) predict returns? Tested on real filing history, no look-ahead.

  python backtest.py            run on data/holdings_history + data/prices (filled by `python refresh.py --history`)
  python backtest.py --selftest synthetic worlds: one where overlap works, one where it does not (the test must tell them apart)

Method. Each month-end T: for every fund use the newest snapshot whose FILING date is on or before T (N-PORT is public ~2 months after the
period). Score each stock by the number of distinct funds holding it (overlap) and by weight increases vs the fund's previous snapshot (flow).
Take the top-N, hold h trading days, compare with the equal-weighted average of every stock any fund holds (the "pool"). Reports mean
excess return, hit rate, rank correlation (IC), and a permutation test: how often do 25 random pool stocks do as well?

Known biases, printed with the results. (1) Fund-selection hindsight: the funds were picked because they did well recently, so the
earlier periods flatter the signal; the honest test is only on periods after a fund was added. (2) Overlapping windows make the
t-statistics too optimistic; read the permutation p-value instead. (3) Prices are today's tickers only: delisted names are missing.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_history(hist_dir):
    """-> {fund: DataFrame(period, filed, weights: Series)} sorted by filing date."""
    out = {}
    for f in sorted(Path(hist_dir).iterdir()) if Path(hist_dir).exists() else []:
        ip = f / "filed.csv"
        if not ip.exists(): continue
        idx = pd.read_csv(ip).sort_values("filed")
        snaps = [(pd.Timestamp(r.filed), str(r.period), pd.read_csv(f / f"{r.period}.csv").set_index("ticker").weight) for r in idx.itertuples()
                 if (f / f"{r.period}.csv").exists()]
        if snaps: out[f.name] = snaps
    return out


def load_prices(price_dir):
    cols = {}
    for p in Path(price_dir).glob("*.csv"):
        d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
        cols[p.stem.upper()] = d[~d.index.duplicated()]
    return pd.DataFrame(cols).sort_index()


def scores_at(hist, T):
    """Overlap and flow score for every stock as of date T using only filings public by T."""
    ov, fl = {}, {}
    for fund, snaps in hist.items():
        vis = [s for s in snaps if s[0] <= T]
        if not vis: continue
        cur = vis[-1][2]; prev = vis[-2][2] if len(vis) > 1 else None
        for t in cur.index: ov[t] = ov.get(t, 0) + 1
        if prev is not None:
            for t in set(cur.index) | set(prev.index):
                a, b = cur.get(t, 0.0), prev.get(t, 0.0)
                if b == 0 and a > 0: fl[t] = fl.get(t, 0) + 1.0        # initiation
                elif a == 0 and b > 0: fl[t] = fl.get(t, 0) - 1.0      # exit
                elif b > 0 and a / b > 1.25: fl[t] = fl.get(t, 0) + 0.5 # clear add (weight also drifts with price, so 25% margin)
                elif b > 0 and a / b < 0.75: fl[t] = fl.get(t, 0) - 0.5
    return pd.Series(ov, dtype=float), pd.Series(fl, dtype=float)


def run(hist, px, horizons=(21, 63, 126), top_n=25, n_perm=500, seed=0, min_pool=40):
    rng = np.random.default_rng(seed)
    dates = px.index; rebal = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).last().values
    res = []
    for T in rebal:
        ov, fl = scores_at(hist, T)
        pool = [t for t in ov.index if t in px.columns]
        if len(pool) < min_pool: continue
        i = dates.get_loc(T)
        for h in horizons:
            if i + h >= len(dates): continue
            r = (px.iloc[i + h][pool] / px.iloc[i][pool] - 1).dropna()
            if len(r) < min_pool: continue
            sc = {"overlap": ov.reindex(r.index).fillna(0), "flow": fl.reindex(r.index).fillna(0)}
            sc["combined"] = sc["overlap"].rank(pct=True) + 0.5 * sc["flow"].rank(pct=True)
            for name, s in sc.items():
                # ties are broken at random: otherwise "first in list" would silently be the tiebreak
                order = (s + rng.random(len(s)) * 1e-6).sort_values(ascending=False)
                pick = order.index[:top_n]
                ex = r[pick].mean() - r.mean()
                ic = s.rank().corr(r.rank()) if s.nunique() > 1 else np.nan
                rand = np.array([r.iloc[rng.choice(len(r), min(top_n, len(r)), replace=False)].mean() - r.mean() for _ in range(n_perm // 10)])
                res.append(dict(date=T, h=h, signal=name, excess=ex, hit=float(r[pick].mean() > r.mean()), ic=ic, p_perm=float((rand >= ex).mean()), n_pool=len(r)))
    return pd.DataFrame(res)


def summarize(res):
    if res.empty: return res
    g = res.groupby(["signal", "h"])
    def t_nonoverlap(x):          # sample every (h/21)th month so windows do not overlap, then an ordinary t-statistic
        h = int(x.name[1]); y = x.sort_index().iloc[:: max(1, round(h / 21))]
        return y.mean() / (y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 2 and y.std(ddof=1) > 0 else np.nan
    ex = res.set_index("date").groupby(["signal", "h"]).excess
    t = pd.Series({k: t_nonoverlap(v.rename(k)) for k, v in ex})
    return pd.DataFrame({"months": g.size(), "mean_excess_%": (100 * g.excess.mean()).round(2), "hit_rate": g.hit.mean().round(2),
                         "mean_IC": g.ic.mean().round(3), "t_nonoverlap": t.round(2), "avg_perm_p": g.p_perm.mean().round(2)})


def verdict(summ, h=63):
    """Plain-language read: a signal only 'works' if excess and IC are positive and the non-overlapping t-statistic is above 1.5."""
    out = {}
    for sig in ["overlap", "flow", "combined"]:
        try: r = summ.loc[(sig, h)]
        except KeyError: continue
        out[sig] = bool(r["mean_excess_%"] > 0 and r["mean_IC"] > 0.02 and r["t_nonoverlap"] > 1.5 and r["months"] >= 12)
    return out


def synthetic(edge, seed=1, n_stocks=150, n_funds=12, months=48):
    """A fake world: stocks have a latent quality that funds partly see (edge) and that drives later returns."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-01", periods=months * 21)
    names = [f"S{i:03d}" for i in range(n_stocks)]
    q = rng.normal(0, 1, n_stocks)
    drift = 0.0006 * q * (1 if edge else 0)
    rets = rng.normal(0.0003, 0.015, (len(days), n_stocks)) + drift
    px = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days, columns=names)
    root = Path(sys.argv[0]).resolve().parent / "_bt_tmp"
    hist = {}
    for f in range(n_funds):
        snaps = []
        for qtr in range(months // 3):
            d = days[min(qtr * 63 + 20, len(days) - 1)]
            view = q + rng.normal(0, 1.0, n_stocks)           # each fund sees quality with noise
            pick = np.argsort(-view)[:30]
            w = pd.Series(rng.uniform(1, 5, 30), index=[names[i] for i in pick])
            snaps.append((d + pd.Timedelta(days=45), str(d.date()), w))
        hist[f"F{f}"] = snaps
    return hist, px


def selftest():
    for edge in (True, False):
        hist, px = synthetic(edge)
        summ = summarize(run(hist, px, n_perm=300))
        v = verdict(summ)
        assert v.get("overlap") is edge, (edge, summ)
    # no look-ahead: a snapshot filed after T must be invisible at T
    w = pd.Series({"A": 1.0})
    ov, _ = scores_at({"F": [(pd.Timestamp("2022-03-01"), "2022-01-31", w)]}, pd.Timestamp("2022-02-28"))
    assert len(ov) == 0
    ov, _ = scores_at({"F": [(pd.Timestamp("2022-03-01"), "2022-01-31", w)]}, pd.Timestamp("2022-03-01"))
    assert ov["A"] == 1
    print("backtest selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest(); sys.exit()
    hist = load_history(ROOT / "data/holdings_history"); px = load_prices(ROOT / "data/prices")
    if not hist or px.empty: sys.exit("No history yet: run `python refresh.py --history` (needs internet) first.")
    summ = summarize(run(hist, px)); print(summ.to_string()); print("\nverdict at 63d:", verdict(summ))
    summ.to_csv(ROOT / "data/backtest_summary.csv")
    print(__doc__.split("Known biases")[1].strip())
