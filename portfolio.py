"""Live track record and concentration check, written to docs/data/track.json and docs/data/risk.json by the daily job.

Track record: every board frozen at the time it was made (each Monday's weekly freeze, plus the two earliest boards) is a cohort.
A cohort's top 25 is bought equal-weight at that day's close and compared with SPY and with the average of every name the model
scored that day (the pool). Nothing is recomputed with hindsight: the cohort files are what the board said then.
The live portfolio holds the latest cohort's top 25, switching on each new cohort date.
Post-mortems: names more than 5 points behind SPY, with what put them on the board and where they rank now.

Concentration: the current top 25 by theme and sector, how correlated their daily returns have been over the past year,
the effective number of independent bets, groups that move together, and the basket's beta to the S&P 500.
"""
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GROUPS = {"Earnings": ["earn_growth", "earn_surprise", "earn_consistency", "earn_revisions", "earn_accel", "earn_sales", "earn_reaction", "earn_guidance"],
          "Manager overlap": ["breadth", "active_weight"], "Manager flow": ["flow"]}


SHADOW = {"Earnings": "earn_score", "Best-ideas overlap": "sig_breadth", "Manager flow": "sig_flow", "Momentum": "sig_momentum"}
FLOW_RULE = ("Flow now runs on daily holdings for part of the fund list. Raise its weight from 5 to 10 points once its own top 25 has beaten "
             "the pool on at least two thirds of weekly boards over three months or more; cut it to 0 if it trails over six months.")


def cohorts():
    """[(date, label, ranking frame with every name scored that day)], oldest first."""
    out = []
    for path, d, lab in [(ROOT / "data_web/ranking_2026_10_15.csv", "2026-10-15", "early board"),
                         (ROOT / "data/daily/2026-10-22.csv", "2026-10-22", "early board")]:
        if path.exists(): out.append((d, lab, pd.read_csv(path)))
    wd = ROOT / "data/weeks"
    if wd.exists():
        for w in sorted(p for p in wd.iterdir() if p.is_dir()):
            f = w / "ranking_all.csv" if (w / "ranking_all.csv").exists() else w / "ranking.csv"
            if f.exists(): out.append((w.name, "weekly board", pd.read_csv(f)))
    out = sorted({d: (d, l, r) for d, l, r in out}.values(), key=lambda x: x[0])
    for _, _, r in out:
        if "rank" not in r.columns: r["rank"] = r["score"].rank(ascending=False, method="first") if "score" in r.columns else np.arange(1, len(r) + 1)
    return out


def _entry_idx(closes, d):
    i = closes.index.searchsorted(pd.Timestamp(d), side="right") - 1      # last close on or before the board date
    return i if i >= 0 else None


def drivers(row):
    pts = {g: sum(float(row.get(f"pts_{k}", 0) or 0) for k in ks) for g, ks in GROUPS.items()}
    if not any(pts.values()): return ""
    top = sorted(pts.items(), key=lambda x: -x[1])[:2]
    return " and ".join(g.lower() for g, _ in top)


def track(closes, latest_rank, today):
    if closes.empty or "SPY" not in closes.columns: return dict(cohorts=[], note="No prices yet.")
    closes = closes.sort_index().ffill(); last = len(closes) - 1; spy = closes["SPY"]
    cur_rank = latest_rank.set_index("ticker")["rank"] if len(latest_rank) else pd.Series(dtype=float)
    res, post = [], []
    for d, lab, r in cohorts():
        i0 = _entry_idx(closes, d)
        if i0 is None or i0 >= last: continue
        top = r[r["rank"] <= 25]; pool = [t for t in r.ticker if t in closes.columns]
        names = [t for t in top.ticker if t in closes.columns]
        if not names: continue
        ret = lambda ts: (closes[ts].iloc[last] / closes[ts].iloc[i0] - 1)
        tr = ret(names); pr = ret(pool); sr = float(spy.iloc[last] / spy.iloc[i0] - 1)
        sig = {}
        for name, col in SHADOW.items():               # each signal's own top 25 that day, to see which signals are earning their weight
            if col in r.columns and r[col].notna().sum() >= 50 and r[col].nunique() > 5:
                pk = [t for t in r.sort_values(col, ascending=False).ticker if t in closes.columns][:25]
                if pk: sig[name] = round(float(ret(pk).mean() - pr.mean()), 4)
        res.append(dict(date=d, label=lab, days=int(last - i0), n=len(names), top25=round(float(tr.mean()), 4), pool=round(float(pr.mean()), 4), spy=round(sr, 4), by_signal=sig,
                        beat_spy=int((tr > sr).sum()), best=[[t, round(float(v), 4)] for t, v in tr.sort_values().tail(3)[::-1].items()],
                        worst=[[t, round(float(v), 4)] for t, v in tr.sort_values().head(3).items()]))
        for t, v in tr.items():
            if v - sr < -0.05:
                row = top[top.ticker == t].iloc[0].to_dict()
                post.append(dict(cohort=d, ticker=t, ret=round(float(v), 4), vs_spy=round(float(v - sr), 4), picked_on=drivers(row),
                                 rank_then=int(row["rank"]), rank_now=(int(cur_rank[t]) if t in cur_rank.index else None)))
    # live portfolio: hold the latest cohort's top 25, switching on each cohort date
    nav = []; cs = cohorts()
    if cs:
        start = _entry_idx(closes, cs[0][0])
        if start is not None:
            v = s = p = 1.0; held = pool = None; ci = 0
            for i in range(start, len(closes)):
                dt = closes.index[i]
                while ci < len(cs) and pd.Timestamp(cs[ci][0]) <= dt:
                    rr = cs[ci][2]; held = [t for t in rr[rr["rank"] <= 25].ticker if t in closes.columns]; pool = [t for t in rr.ticker if t in closes.columns]; ci += 1
                if i > start and held:
                    g = closes.iloc[i] / closes.iloc[i - 1] - 1
                    v *= 1 + float(g[held].mean()); p *= 1 + float(g[pool].mean()); s *= 1 + float(g["SPY"])
                nav.append([str(dt.date()), round(v, 4), round(s, 4), round(p, 4)])
    post.sort(key=lambda x: x["vs_spy"])
    card = []
    for name in SHADOW:
        v = [c["by_signal"][name] for c in res if name in c.get("by_signal", {})]
        if v: card.append(dict(signal=name, boards=len(v), avg_vs_pool=round(float(np.mean(v)), 4), beat_pool=int(sum(x > 0 for x in v))))
    return dict(cohorts=res[::-1], nav=nav, postmortems=post[:30], scorecard=card, flow_rule=FLOW_RULE, asof=str(today),
                note="Equal weight, no costs, prices to the latest close. Early boards were made while the model was still being built.")


def concentration(rk, closes, auto, today):
    top = rk[rk["rank"] <= 25].copy()
    if top.empty: return dict(note="No board yet.")
    n = len(top)
    themes = top["bucket"].fillna("Other").value_counts() if "bucket" in top.columns else pd.Series(dtype=int)
    sec = auto.reindex(top.ticker)["sector"].fillna("Unknown") if "sector" in auto.columns else pd.Series("Unknown", index=top.ticker)
    out = dict(asof=str(today), n=n, themes=[[k, int(v), round(v / n, 3)] for k, v in themes.items()],
               sectors=[[k, int(v), round(v / n, 3)] for k, v in sec.value_counts().items()])
    names = [t for t in top.ticker if t in closes.columns]
    if len(names) >= 5 and "SPY" in closes.columns:
        r = closes[names + ["SPY"]].iloc[-253:].pct_change().iloc[1:]
        r = r.loc[:, r.notna().mean() > 0.8].fillna(0)
        names = [c for c in r.columns if c != "SPY"]
        c = r[names].corr(); m = c.values; k = len(names)
        avg = float((m.sum() - k) / (k * (k - 1)))
        ev = np.clip(np.linalg.eigvalsh(m), 0, None); eff = float(ev.sum() ** 2 / (ev ** 2).sum())
        basket = r[names].mean(axis=1); beta = float(np.cov(basket, r["SPY"])[0, 1] / r["SPY"].var())
        # groups that move together: connected components at correlation >= 0.6
        seen, groups = set(), []
        for t in names:
            if t in seen: continue
            stack, comp = [t], []
            while stack:
                x = stack.pop()
                if x in seen: continue
                seen.add(x); comp.append(x); stack += [y for y in names if y not in seen and c.loc[x, y] >= 0.6]
            if len(comp) >= 2: groups.append(sorted(comp))
        roll = basket.rolling(20).sum()
        pairs = sorted(((c.loc[a, b], a, b) for i, a in enumerate(names) for b in names[i + 1:]), reverse=True)[:8]
        out.update(avg_corr=round(avg, 3), effective_bets=round(eff, 1), beta=round(beta, 2), groups=sorted(groups, key=len, reverse=True),
                   largest_group_share=round(max((len(g) for g in groups), default=1) / n, 3),
                   worst_20d=round(float(roll.min()), 4), vol_ann=round(float(basket.std() * np.sqrt(252)), 4),
                   spy_vol_ann=round(float(r["SPY"].std() * np.sqrt(252)), 4), top_pairs=[[a, b, round(float(v), 2)] for v, a, b in pairs])
    return out


def write(rk, closes, auto, today):
    d = ROOT / "docs/data"; d.mkdir(parents=True, exist_ok=True)
    t = track(closes, rk, today); (d / "track.json").write_text(json.dumps(t, default=str))
    k = concentration(rk, closes, auto, today); (d / "risk.json").write_text(json.dumps(k, default=str))
    return f"track record: {len(t.get('cohorts', []))} boards tracked; concentration: {k.get('effective_bets', '?')} effective bets, avg correlation {k.get('avg_corr', '?')}"


def selftest():
    idx = pd.bdate_range("2026-01-01", periods=300)
    rng = np.random.default_rng(0); base = rng.normal(0, 0.01, (300, 1))
    cl = pd.DataFrame(100 * np.exp(np.cumsum(base + rng.normal(0, 0.005, (300, 6)), axis=0)), index=idx, columns=["A", "B", "C", "D", "E", "SPY"])
    rk = pd.DataFrame({"ticker": list("ABCDE"), "rank": [1, 2, 3, 4, 5], "bucket": ["AI", "AI", "AI", "Fin", "Other"]})
    k = concentration(rk, cl, pd.DataFrame({"sector": ["Tech"] * 5}, index=list("ABCDE")), date(2026, 10, 29))
    assert k["n"] == 5 and 0.5 < k["avg_corr"] < 1 and 1 <= k["effective_bets"] < 3 and k["themes"][0] == ["AI", 3, 0.6], k
    assert drivers({"pts_earn_growth": 10, "pts_breadth": 20, "pts_flow": 1}) == "manager overlap and earnings"
    print("portfolio selftest ok")


if __name__ == "__main__":
    selftest()
