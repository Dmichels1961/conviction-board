"""Paper portfolio: the 5 highest-conviction names from the main board, rebalanced every 60 days. No real money.

Rules (fixed at the start, 29 October 2026):
  - Picks: the 5 highest-ranked names on the main board, no sector cap. Equal weight (20% each).
    (Backtest 2021-26 of these rules: 20.2%/yr vs 12.1% S&P 500 and 14.8% QQQ, with high beta. The risk-controlled board's top 10
    returned 8.9%/yr: its edge is small and spread over many names, so 10 names cannot express it.)
  - Bought at the day's close.
  - Every 60 calendar days: sell names that are no longer picked, buy the new picks, reset all to equal weight.
  - Costs: 0.10% of every trade's value (spread and slippage). Taxes estimated at 37% on short-term realized gains, shown, not deducted.
  - Starts with $100,000. Compared with the S&P 500 (SPY) and Nasdaq-100 (QQQ) from the same start, dividends included in all three.
State lives in data/paper/state.json; the daily job marks it to market and rebalances on schedule.
"""
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "data/paper/state.json"
N, MAX_PER_SECTOR, HOLD_DAYS, COST, TAX, START_CASH = 5, None, 60, 0.001, 0.37, 100_000.0    # 5 names, no sector cap (owner's choice)


def picks(rk, n=N, cap=MAX_PER_SECTOR):
    b = rk.sort_values("rank")
    out, per = [], {}
    for _, r in b.iterrows():
        s = r.get("sector") if isinstance(r.get("sector"), str) else "Unknown"
        if cap and s != "Unknown" and per.get(s, 0) >= cap: continue
        out.append(r["ticker"]); per[s] = per.get(s, 0) + 1
        if len(out) == n: break
    for t in b.ticker:                       # only if the cap leaves seats empty: fill them in rank order
        if len(out) >= n: break
        if t not in out: out.append(t)
    return out


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True); STATE.write_text(json.dumps(st, indent=1))


def _value(st, px):
    return st["cash"] + sum(p["shares"] * px.get(t, p["last"]) for t, p in st["positions"].items())


def rebalance(st, targets, px, day, log=print):
    """Sell what is not a target, then bring every target to equal weight. Trades at the close, cost on each trade's value."""
    for t, p in list(st["positions"].items()):
        p["last"] = float(px.get(t, p["last"]))
    nav = _value(st, px); target_val = nav * (1 - COST) / len(targets)
    trades, realized = [], 0.0
    for t in list(st["positions"]):
        if t not in targets:
            p = st["positions"].pop(t); val = p["shares"] * p["last"]; gain = val - p["shares"] * p["cost_basis"]
            st["cash"] += val * (1 - COST); realized += gain
            trades.append(dict(date=day, ticker=t, action="sell", shares=round(p["shares"], 4), price=round(p["last"], 2), value=round(val, 2), gain=round(gain, 2)))
    for t in targets:
        price = float(px[t]); have = st["positions"].get(t)
        cur = have["shares"] * price if have else 0.0; diff = target_val - cur
        if abs(diff) < 1: continue
        sh = diff / price
        if diff > 0:
            st["cash"] -= diff * (1 + COST)
            if have: have["cost_basis"] = (have["shares"] * have["cost_basis"] + sh * price) / (have["shares"] + sh); have["shares"] += sh
            else: st["positions"][t] = dict(shares=sh, cost_basis=price, entry_date=day, last=price)
        else:
            gain = -sh * (price - have["cost_basis"]); realized += gain
            have["shares"] += sh; st["cash"] += -diff * (1 - COST)
        trades.append(dict(date=day, ticker=t, action="buy" if diff > 0 else "trim", shares=round(abs(sh), 4), price=round(price, 2), value=round(abs(diff), 2),
                           gain=None if diff > 0 else round(gain, 2)))
    st["trades"] += trades; st["realized"] = st.get("realized", 0.0) + realized
    st["costs"] = st.get("costs", 0.0) + sum(x["value"] for x in trades) * COST
    st["last_rebalance"] = day; st["next_rebalance"] = str(date.fromisoformat(day) + timedelta(days=HOLD_DAYS))
    log(f"paper portfolio: rebalanced on {day}, {len(trades)} trades, holdings {', '.join(targets)}")


def update(rk, closes, today, log=print, trade=True):
    """Mark to market; start or rebalance when due. Writes docs/data/paper.json."""
    if closes.empty: log("paper portfolio: no prices"); return None
    last_day = str(closes.index[-1].date()); px = closes.ffill().iloc[-1].dropna().to_dict()   # latest price per name, even if one feed has not printed the last day
    st = load()
    if st is None:
        targets = [t for t in picks(rk) if t in px]
        st = dict(start=last_day, cash=START_CASH, positions={}, trades=[], nav=[], realized=0.0, costs=0.0,
                  bench_start={b: float(px[b]) for b in ("SPY", "QQQ") if b in px})
        rebalance(st, targets, px, last_day, log)
    elif trade and last_day >= st["next_rebalance"] and st.get("last_rebalance") != last_day:   # trades only on the evening run, at the close
        targets = [t for t in picks(rk) if t in px]
        if len(targets) >= N // 2: rebalance(st, targets, px, last_day, log)
    for t, p in st["positions"].items(): p["last"] = float(px.get(t, p["last"]))
    nav = _value(st, px); bs = st["bench_start"]
    row = [last_day, round(nav, 2)] + [round(START_CASH * px[b] / bs[b], 2) if b in px and b in bs else None for b in ("SPY", "QQQ")]
    st["nav"] = [r for r in st["nav"] if r[0] != last_day] + [row]
    save(st); write_site(st, rk, last_day)
    log(f"paper portfolio: ${nav:,.0f} ({nav / START_CASH - 1:+.1%}); S&P {row[2]} QQQ {row[3]}; next rebalance {st['next_rebalance']}")
    return st


def write_site(st, rk, day):
    navs = st["nav"]; nav = navs[-1][1]
    vals = pd.Series([r[1] for r in navs]); dd = float((vals / vals.cummax() - 1).min()) if len(vals) else 0.0
    name = rk.set_index("ticker")["name"].to_dict() if "name" in rk.columns else {}
    sector = rk.set_index("ticker")["sector"].to_dict() if "sector" in rk.columns else {}
    rcr = rk.set_index("ticker")["rank"].to_dict() if "rank" in rk.columns else {}
    hold = []
    for t, p in st["positions"].items():
        v = p["shares"] * p["last"]
        hold.append(dict(ticker=t, name=name.get(t), sector=sector.get(t) if isinstance(sector.get(t), str) else None, weight=round(v / nav, 4),
                         value=round(v, 2), cost=round(p["cost_basis"], 2), price=round(p["last"], 2), ret=round(p["last"] / p["cost_basis"] - 1, 4),
                         since=p["entry_date"], rank_now=int(rcr[t]) if t in rcr and pd.notna(rcr[t]) else None))
    hold.sort(key=lambda h: -h["value"])
    unreal = sum(p["shares"] * (p["last"] - p["cost_basis"]) for p in st["positions"].values())
    out = dict(asof=day, start=st["start"], nav=round(nav, 2), ret=round(nav / START_CASH - 1, 4),
               spy=navs[-1][2], qqq=navs[-1][3], ret_spy=round(navs[-1][2] / START_CASH - 1, 4) if navs[-1][2] else None,
               ret_qqq=round(navs[-1][3] / START_CASH - 1, 4) if navs[-1][3] else None, max_drawdown=round(dd, 4),
               next_rebalance=st["next_rebalance"], days_to_rebalance=(date.fromisoformat(st["next_rebalance"]) - date.fromisoformat(day)).days,
               cash=round(st["cash"], 2), costs=round(st.get("costs", 0.0), 2), realized=round(st.get("realized", 0.0), 2),
               est_tax=round(max(0.0, st.get("realized", 0.0)) * TAX, 2), unrealized=round(unreal, 2), holdings=hold,
               trades=st["trades"][-40:][::-1], series=navs,
               rules="The 5 highest-ranked names on the main board, 20% each, rebalanced every 60 days. "
                     "0.10% cost per trade. Taxes estimated at 37% on realized short-term gains (shown, not deducted). No real money.")
    p = ROOT / "data/paper/backtest.json"
    if p.exists(): out["backtest"] = json.loads(p.read_text())
    (ROOT / "docs/data").mkdir(parents=True, exist_ok=True); (ROOT / "docs/data/paper.json").write_text(json.dumps(out))


def selftest():
    import tempfile, shutil
    global STATE, ROOT
    old = (STATE, ROOT); ROOT = Path(tempfile.mkdtemp()); STATE = ROOT / "data/paper/state.json"
    try:
        idx = pd.bdate_range("2026-10-29", periods=70)
        names = [f"S{i}" for i in range(14)]
        cl = pd.DataFrame({t: 100 + np.arange(70) * (i - 5) * 0.1 for i, t in enumerate(names)}, index=idx)
        cl["SPY"] = 400.0; cl["QQQ"] = 500.0
        rk = pd.DataFrame({"ticker": names, "rank": range(1, 15), "sector": ["Tech"] * 5 + ["Energy"] * 3 + ["Health"] * 3 + ["Banks"] * 3, "name": names})
        st = update(rk, cl.iloc[:1], idx[0].date(), log=lambda *a: None)
        assert set(st["positions"]) == set(names[:N])                                         # top N by rank, no cap
        assert picks(rk, cap=3).count("S3") == 0                                              # the cap still works when asked for
        assert abs(_value(st, cl.iloc[0].to_dict()) - START_CASH * (1 - COST)) < 50
        rk2 = rk.assign(rank=list(range(14, 0, -1)))                                     # ranking reversed
        st = update(rk2, cl.iloc[:30], idx[29].date(), log=lambda *a: None); assert st["last_rebalance"] == str(idx[0].date())   # not due yet
        st = update(rk2, cl.iloc[:45], idx[44].date(), log=lambda *a: None); assert st["last_rebalance"] == str(idx[44].date())  # 60 days passed
        assert set(st["positions"]) == set(picks(rk2)) and len(st["positions"]) == N and len(st["nav"]) == 3
    finally:
        shutil.rmtree(ROOT); STATE, ROOT = old
    print("paper selftest ok")


if __name__ == "__main__":
    selftest()
