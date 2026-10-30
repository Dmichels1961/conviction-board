"""Machine-pulled inputs for every candidate (names held by 2+ funds), refreshed by the daily job. Free sources only.

data_web/auto.csv holds one row per ticker. The scorer overlays it on the hand-collected earnings.csv, so a value pulled
here always wins and names without hand data still get scored. Each column records how fresh it is via a *_asof column.

  Source and cadence (every source is free and needs no key):
    daily, whole pool   prices / momentum, consensus EPS + growth + revisions, revenue growth      Yahoo (yfinance)
    when due            earnings surprises + next report date (new, week old, or just reported)    Yahoo
    daily, from above   market reaction after the last 4 reports (vs SPY)                          computed
    every ~3 days       short interest and its change, beta, forward P/E                           Yahoo quote summary
    weekly per name     gross profitability, accruals, asset growth, net share issuance            SEC EDGAR XBRL
    every ~5 days       distinct insiders buying in the open market, last 90 days                  SEC Form 4
"""
import json, time
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AUTO = ROOT / "data_web/auto.csv"
STATE = ROOT / "data/auto_state.json"


# ----------------------------------------------------------------- storage
def load_auto(pool=()):
    a = pd.read_csv(AUTO).set_index("ticker") if AUTO.exists() else pd.DataFrame(index=pd.Index([], name="ticker"))
    new = [t for t in pool if t not in a.index]
    if new: a = pd.concat([a, pd.DataFrame(index=pd.Index(new, name="ticker"))])
    return a


def save_auto(a):
    AUTO.parent.mkdir(parents=True, exist_ok=True)
    a.sort_index().reset_index().to_csv(AUTO, index=False)


def setv(a, t, col, v):
    if col not in a.columns: a[col] = pd.Series(np.nan, index=a.index, dtype=object if isinstance(v, str) else float)
    elif isinstance(v, str) and a[col].dtype != object: a[col] = a[col].astype(object)
    a.loc[t, col] = v


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True); STATE.write_text(json.dumps(s, indent=0, sort_keys=True))


def due(pool, last_done, today, every_days, cap, a=None, mark=None):
    """Oldest-first rotation: names never done or done `every_days`+ ago, at most `cap` per run.
    With `a` and `mark` (a column such as 'info_asof'), a name the state calls done but whose data is missing is due again:
    the state file and the data file can disagree after an interrupted or overlapping run."""
    if a is not None and mark:
        have = set(a.index[a[mark].notna()]) if mark in a.columns else set()
        last_done = {t: d for t, d in last_done.items() if t in have}
    age = lambda t: (pd.Timestamp(today) - pd.Timestamp(last_done[t])).days if t in last_done else 10 ** 6
    return sorted([t for t in pool if age(t) >= every_days], key=age, reverse=True)[:cap]


# ----------------------------------------------------------------- Yahoo quote summary (short interest, beta, P/E)
def info_fields(info):
    g = lambda k: (float(info[k]) if isinstance(info.get(k), (int, float)) and not isinstance(info.get(k), bool) else np.nan)
    sp, s_now, s_prev = g("shortPercentOfFloat"), g("sharesShort"), g("sharesShortPriorMonth")
    return dict(short_pct=sp, short_chg=(s_now / s_prev - 1) if s_now > 0 and s_prev > 0 else np.nan,
                beta=g("beta"), fwd_pe=g("forwardPE"), pt_mean=g("targetMeanPrice"),
                name=info.get("shortName") or info.get("longName"), sector=info.get("sector"), industry=info.get("industry"))


def refresh_info(a, pool, today, log=print, cap=420, fetch=None, sleep=0.35):
    if fetch is None:
        import yfinance as yf
        fetch = lambda t: yf.Ticker(t).info
    st = load_state(); done = st.setdefault("info", {}); ok = 0
    for t in due(pool, done, today, 3, cap, a, "info_asof"):
        try:
            f = info_fields(fetch(t))
            for k, v in f.items():
                if (isinstance(v, float) and not np.isnan(v)) or (isinstance(v, str) and v): setv(a, t, k, v)
            setv(a, t, "info_asof", str(today)); done[t] = str(today); ok += 1
        except Exception as e:
            log(f"info {t}: {str(e)[:80]}")
        if sleep: time.sleep(sleep)
    save_state(st); log(f"short interest / beta / P/E (Yahoo): {ok} names refreshed")


# ----------------------------------------------------------------- SEC XBRL fundamentals
CONCEPTS = {
    "assets": [("us-gaap", "Assets"), ("ifrs-full", "Assets")],
    "gross_profit": [("us-gaap", "GrossProfit"), ("ifrs-full", "GrossProfit")],
    "revenue": [("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"), ("us-gaap", "Revenues"), ("us-gaap", "SalesRevenueNet"),
                ("ifrs-full", "Revenue")],
    "cogs": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold"), ("us-gaap", "CostOfGoodsSold"), ("ifrs-full", "CostOfSales")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"), ("ifrs-full", "ProfitLoss")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"), ("ifrs-full", "CashFlowsFromUsedInOperatingActivities")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding")],
    "rd": [("us-gaap", "ResearchAndDevelopmentExpense"), ("us-gaap", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"),
           ("ifrs-full", "ResearchAndDevelopmentExpense")],
    "capex": [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"), ("us-gaap", "PaymentsToAcquireProductiveAssets"),
              ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities")],
}
FORMS = {"10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A"}


def _facts(cf, concept):
    """All facts for the first concept alias that has data -> DataFrame(start, end, val, filed). One unit only (ratios need no FX)."""
    for tax, name in CONCEPTS[concept]:
        node = cf.get("facts", {}).get(tax, {}).get(name)
        if not node: continue
        units = node.get("units", {})
        unit = "shares" if "shares" in units else next(iter(units), None)
        rows = [x for x in units.get(unit, []) if x.get("form") in FORMS]
        if not rows: continue
        d = pd.DataFrame(rows)
        d["end"] = pd.to_datetime(d["end"]); d["start"] = pd.to_datetime(d.get("start"), errors="coerce")
        d["filed"] = pd.to_datetime(d["filed"])
        return d.sort_values("filed").drop_duplicates(["start", "end"], keep="last")[["start", "end", "val", "filed"]]
    return pd.DataFrame(columns=["start", "end", "val", "filed"])


def ttm(d, asof=None):
    """Trailing 12 months from annual + year-to-date figures: last fiscal year + latest YTD - same YTD a year earlier.
    With asof, only periods ending on or before that date count (used for the year-earlier comparison)."""
    d = d.dropna(subset=["start"])
    if asof is not None: d = d[d.end <= asof]
    if d.empty: return np.nan, None
    d = d.assign(days=(d.end - d.start).dt.days)
    ann = d[d.days.between(350, 380)].sort_values("end")
    if ann.empty: return np.nan, None
    A = ann.iloc[-1]
    ytd = d[d.days.between(80, 300) & (d.end > A.end)].sort_values(["end", "days"])
    if len(ytd):
        Y = ytd.iloc[-1]
        P = d[((d.end - (Y.end - pd.DateOffset(years=1))).dt.days.abs() <= 10) & ((d.days - Y.days).abs() <= 10)]
        if len(P): return float(A.val + Y.val - P.iloc[-1].val), Y.end
    return float(A.val), A.end


def instant_now_and_year_ago(d, sum_same_end=False):
    if d.empty: return np.nan, np.nan, None
    g = d.groupby("end").val.sum() if sum_same_end else d.groupby("end").val.last()
    g = g.sort_index(); end = g.index[-1]
    old = g[(g.index <= end - pd.Timedelta(days=300)) & (g.index >= end - pd.Timedelta(days=430))]
    return float(g.iloc[-1]), (float(old.iloc[-1]) if len(old) else np.nan), end


def fundamentals(cf):
    """companyfacts JSON -> gross profitability, accruals, asset growth, net share issuance (all unit-free ratios)."""
    a_now, a_old, a_end = instant_now_and_year_ago(_facts(cf, "assets"))
    gp, gp_end = ttm(_facts(cf, "gross_profit"))
    if np.isnan(gp):
        rev, r_end = ttm(_facts(cf, "revenue")); cogs, c_end = ttm(_facts(cf, "cogs"))
        if not np.isnan(rev) and not np.isnan(cogs) and r_end is not None and c_end is not None and abs((r_end - c_end).days) <= 10: gp, gp_end = rev - cogs, r_end
    ni, _ = ttm(_facts(cf, "net_income")); cfo, _ = ttm(_facts(cf, "cfo"))
    sh_now, sh_old, _ = instant_now_and_year_ago(_facts(cf, "shares"), sum_same_end=True)
    cx = _facts(cf, "capex"); cx_now, cx_end = ttm(cx)
    cx_old = ttm(cx, cx_end - pd.Timedelta(days=330))[0] if cx_end is not None else np.nan
    rvf = _facts(cf, "revenue"); rv_now = ttm(rvf)[0]
    rv_old = ttm(rvf, cx_end - pd.Timedelta(days=330))[0] if cx_end is not None else np.nan
    avg_assets = np.nanmean([a_now, a_old]) if not np.isnan(a_now) else np.nan
    ok = lambda x: x is not None and not (isinstance(x, float) and np.isnan(x))
    return dict(
        gross_prof=gp / a_now if ok(gp) and ok(a_now) and a_now > 0 else np.nan,
        accruals=(ni - cfo) / avg_assets if ok(ni) and ok(cfo) and ok(avg_assets) and avg_assets > 0 else np.nan,
        asset_growth=a_now / a_old - 1 if ok(a_now) and ok(a_old) and a_old > 0 else np.nan,
        net_issuance=sh_now / sh_old - 1 if ok(sh_now) and ok(sh_old) and sh_old > 0 else np.nan,
        capex_growth=abs(cx_now) / abs(cx_old) - 1 if ok(cx_now) and ok(cx_old) and cx_old != 0 else np.nan,
        capex_intensity=abs(cx_now) / rv_now if ok(cx_now) and ok(rv_now) and rv_now > 0 else np.nan,
        capex_intensity_chg=(abs(cx_now) / rv_now - abs(cx_old) / rv_old) if ok(cx_now) and ok(cx_old) and ok(rv_now) and ok(rv_old) and rv_now > 0 and rv_old > 0 else np.nan,
        fin_period=str(a_end.date()) if a_end is not None else "")


def cik_map():
    import edgar as EG
    p = ROOT / "data/sec_tickers.json"
    if not p.exists() or time.time() - p.stat().st_mtime > 7 * 86400:
        p.write_text(EG._get("https://www.sec.gov/files/company_tickers.json"))
    j = json.loads(p.read_text())
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in j.values()}


def refresh_sec(a, pool, today, log=print, cap=300, budget_s=1500, fetch=None):
    import edgar as EG
    try: cmap = cik_map()
    except Exception as e: log(f"SEC ticker map FAILED ({e}); fundamentals skipped"); return
    fetch = fetch or (lambda cik: json.loads(EG._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")))
    st = load_state(); done = st.setdefault("sec", {}); ok = miss = 0; t0 = time.time()
    for t in due(pool, done, today, 7, cap, a, "sec_asof"):
        if time.time() - t0 > budget_s: log("SEC fundamentals: time budget reached, rest continue tomorrow"); break
        cik = cmap.get(t.upper())
        if not cik: done[t] = str(today); miss += 1; continue          # no SEC filer (e.g. a foreign ordinary share)
        try:
            cf = fetch(cik); f = fundamentals(cf)
            try:
                import emerging; f.update(emerging.growth_features(cf))         # R&D, revenue acceleration, margin change
            except Exception as e: log(f"sec {t} growth features: {str(e)[:60]}")
            for k, v in f.items():
                if isinstance(v, str): setv(a, t, k, v)
                elif not np.isnan(v): setv(a, t, k, round(float(v), 5))
            setv(a, t, "sec_asof", str(today)); done[t] = str(today); ok += 1
        except Exception as e:
            log(f"sec {t}: {str(e)[:80]}")
        time.sleep(0.12)                                                  # SEC asks for <= 10 requests a second
    save_state(st); log(f"SEC fundamentals: {ok} names refreshed, {miss} not SEC filers")


# ----------------------------------------------------------------- AI capex gauge
HYPERSCALERS = ["MSFT", "GOOGL", "AMZN", "META", "ORCL"]


def capex_gauge(today, log=print, fetch=None):
    """Trailing-12-month capital spending of the five largest AI-infrastructure buyers, quarter by quarter, from SEC filings.
    Shown on the site as the leading indicator for AI-hardware suppliers; not part of either frozen board's score."""
    import edgar as EG
    fetch = fetch or (lambda cik: json.loads(EG._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")))
    cmap = cik_map(); comp = {}
    for t in HYPERSCALERS:
        try:
            cx = _facts(fetch(cmap[t]), "capex"); ends = sorted(set(cx.end[cx.end >= pd.Timestamp(today) - pd.Timedelta(days=3 * 365)]))
            series = {}
            for e in ends:
                v, ve = ttm(cx, e)
                if v == v and ve is not None and ve == e: series[str(e.date())] = round(abs(v) / 1e9, 1)
            comp[t] = series
        except Exception as e:
            log(f"capex gauge {t}: {str(e)[:60]}")
        time.sleep(0.15)
    rows = []
    for t, s in comp.items():
        if len(s) < 2: continue
        k = sorted(s); last = s[k[-1]]
        cand = [(abs((pd.Timestamp(k[-1]) - pd.Timestamp(x)).days - 365), x) for x in k if 300 <= (pd.Timestamp(k[-1]) - pd.Timestamp(x)).days <= 430]
        yago = [s[min(cand)[1]]] if cand else []
        prev = s[k[-2]]
        rows.append(dict(ticker=t, period=k[-1], ttm_bn=last, yoy=round(last / yago[-1] - 1, 3) if yago else None,
                         qoq=round(last / prev - 1, 3), series=[[x, s[x]] for x in k]))
    tot = sum(r["ttm_bn"] for r in rows); tot_y = [r for r in rows if r["yoy"] is not None]
    out = dict(asof=str(today), companies=rows, total_bn=round(tot, 1),
               total_yoy=round(sum(r["ttm_bn"] for r in tot_y) / sum(r["ttm_bn"] / (1 + r["yoy"]) for r in tot_y) - 1, 3) if tot_y else None,
               note="Trailing-12-month purchases of property and equipment, from each company's 10-Q/10-K. Spending growth is revenue for "
                    "AI-hardware suppliers; a slowdown is the main risk to the AI-heavy part of the boards.")
    (ROOT / "docs/data").mkdir(parents=True, exist_ok=True); (ROOT / "docs/data/capex.json").write_text(json.dumps(out))
    log(f"AI capex gauge: ${out['total_bn']}bn trailing 12 months, {out['total_yoy']} year on year")
    return out


# ----------------------------------------------------------------- market reaction to the last 4 reports
def reactions(closes, events, today, n=4):
    """Average 2-day move vs SPY from the close before each report to the close after it (covers before-open and after-close reports)."""
    if "SPY" not in closes.columns: return {}
    ev = events.copy(); ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.normalize()
    ev = ev[(ev.date <= pd.Timestamp(today)) & (ev.date >= pd.Timestamp(today) - pd.Timedelta(days=420))]
    idx = closes.index; out = {}
    for t, g in ev.groupby("ticker"):
        if t not in closes.columns: continue
        moves = []
        for d in sorted(g.date.unique())[-n:]:
            i = idx.searchsorted(d)                                       # first trading day on/after the report date
            if i < 1 or i + 1 >= len(idx): continue
            p0, p1 = closes[t].iloc[i - 1], closes[t].iloc[i + 1]; s0, s1 = closes["SPY"].iloc[i - 1], closes["SPY"].iloc[i + 1]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0 and s0 > 0: moves.append((p1 / p0 - 1) - (s1 / s0 - 1))
        if moves: out[t] = float(np.mean(moves))
    return out


# ----------------------------------------------------------------- insider buying (SEC Form 4)
def refresh_insiders(a, pool, today, log=print, cap=140, budget_s=600):
    import form4 as F4
    st = load_state(); done = st.setdefault("form4", {}); todo = due(pool, done, today, 5, cap)
    cache = pd.read_csv(F4.CACHE) if F4.CACHE.exists() else pd.DataFrame(columns=F4.COLS); t0 = time.time(); n = 0
    for t in todo:
        if time.time() - t0 > budget_s: log("insiders: time budget reached, rest continue tomorrow"); break
        try:
            new = F4.fetch_new(t, cache, per_ticker=20)
            if len(new): cache = pd.concat([cache, new], ignore_index=True)
            done[t] = str(today); n += 1
        except Exception as e:
            log(f"form4 {t}: {str(e)[:80]}")
    F4.CACHE.parent.mkdir(exist_ok=True); cache.to_csv(F4.CACHE, index=False); save_state(st)
    buyers = F4.buyers_from_cache(cache, today)
    checked = [t for t in pool if t in done]
    for t in checked: setv(a, t, "insider_buyers", float(buyers.get(t, 0)))
    log(f"insiders (SEC Form 4): {n} names checked today, {len(checked)} covered, {int((buyers > 0).sum())} with open-market buying")


# ----------------------------------------------------------------- self-test (offline)
def selftest():
    def fact(s, e, v, f="2026-08-01"): return dict(start=s, end=e, val=v, filed=f, form="10-Q")
    cf = {"facts": {
        "us-gaap": {
            "Assets": {"units": {"USD": [dict(end="2025-06-30", val=100, filed="2025-08-01", form="10-Q"), dict(end="2026-06-30", val=120, filed="2026-08-01", form="10-Q")]}},
            "GrossProfit": {"units": {"USD": [dict(fact("2025-01-01", "2025-12-31", 40), form="10-K"), fact("2026-01-01", "2026-06-30", 24), fact("2025-01-01", "2025-06-30", 18)]}},
            "NetIncomeLoss": {"units": {"USD": [dict(fact("2025-01-01", "2025-12-31", 20), form="10-K")]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [dict(fact("2025-01-01", "2025-12-31", 25), form="10-K")]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            dict(end="2025-07-20", val=50, filed="2025-08-01", form="10-Q"), dict(end="2025-07-20", val=50, filed="2025-08-01", form="10-Q"),
            dict(end="2026-07-20", val=48, filed="2026-08-01", form="10-Q")]}}}}}
    f = fundamentals(cf)
    assert abs(f["gross_prof"] - (40 + 24 - 18) / 120) < 1e-9, f          # TTM = FY + YTD - prior YTD
    assert abs(f["asset_growth"] - 0.2) < 1e-9 and abs(f["accruals"] - (20 - 25) / 110) < 1e-9, f
    assert abs(f["net_issuance"] - (48 / 50 - 1)) < 1e-9, f                # duplicate same-day fact counted once
    idx = pd.bdate_range("2026-01-01", periods=60)
    cl = pd.DataFrame({"X": 100.0, "SPY": 100.0}, index=idx); cl.loc[idx[21]:, "X"] = 110.0; cl.loc[idx[21]:, "SPY"] = 101.0
    r = reactions(cl, pd.DataFrame({"ticker": ["X"], "date": [str(idx[20].date())]}), date(2026, 3, 30))
    assert abs(r["X"] - 0.09) < 1e-9, r
    assert due(["A", "B", "C"], {"A": "2026-10-28", "B": "2026-10-20"}, date(2026, 10, 29), 3, 5) == ["C", "B"]
    assert info_fields({"shortPercentOfFloat": 0.05, "sharesShort": 110, "sharesShortPriorMonth": 100, "beta": True})["short_chg"] - 0.1 < 1e-9
    print("autodata selftest ok")


if __name__ == "__main__":
    selftest()
